"""Rectified-flow training and second-order UniPC inference scheduler."""

from __future__ import annotations

import torch


def _shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


class FlowUniPCScheduler:

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_karras_sigmas: bool = True,
        rho: float = 7.0,
        solver_order: int = 2,
        lower_order_final: bool = True,
        time_distribution: str = "logitnormal",
        training_weight_method: str = "uniform",
    ):
        if solver_order < 1:
            raise ValueError("solver_order must be positive.")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.use_karras_sigmas = bool(use_karras_sigmas)
        self.rho = float(rho)
        self.solver_order = int(solver_order)
        self.lower_order_final = bool(lower_order_final)
        self.time_distribution = str(time_distribution)
        if self.time_distribution not in {"logitnormal", "uniform"}:
            raise ValueError("time_distribution must be 'logitnormal' or 'uniform'.")
        self.training_weight_method = str(training_weight_method)
        if self.training_weight_method != "uniform":
            raise ValueError("training_weight_method must be 'uniform'.")

        alphas = torch.linspace(
            1.0 / self.num_train_timesteps,
            1.0,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        training_sigmas = _shift_sigma(1.0 - alphas, self.shift)
        self._sigma_max = float(training_sigmas[0])
        self._sigma_min = float(training_sigmas[-1])
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)
        self._step_index = 0
        self._model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self._lower_order_nums = 0
        self._last_sample: torch.Tensor | None = None
        self._this_order = 1
        self._stream_states: dict[str, tuple] = {}

    def set_timesteps(
        self,
        num_inference_steps: int,
        device,
        sigma_shift: float | None = None,
    ) -> torch.Tensor:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        shift = self.shift if sigma_shift is None else float(sigma_shift)
        if self.use_karras_sigmas:
            # EDM sigma 200 -> 0.01, rho=7, mapped to rectified-flow time.
            ramp = torch.arange(num_inference_steps + 1, dtype=torch.float64) / num_inference_steps
            maximum = 200.0 ** (1.0 / self.rho)
            minimum = 0.01 ** (1.0 / self.rho)
            edm_sigma = (maximum + ramp * (minimum - maximum)).pow(self.rho)
            inference_sigmas = edm_sigma / (1.0 + edm_sigma)
        else:
            inference_sigmas = torch.linspace(
                self._sigma_max,
                self._sigma_min,
                num_inference_steps + 1,
                dtype=torch.float64,
            )[:-1]
            inference_sigmas = _shift_sigma(inference_sigmas, shift)

        self.sigmas = torch.cat((inference_sigmas.float(), torch.zeros(1)))
        self.timesteps = inference_sigmas.to(device=device, dtype=torch.float32)
        self._step_index = 0
        self._model_outputs = [None] * self.solver_order
        self._lower_order_nums = 0
        self._last_sample = None
        self._this_order = 1
        self._stream_states.clear()
        return self.timesteps

    def _restore_stream_state(self, state: tuple | None) -> None:
        if state is None:
            self._step_index = 0
            self._model_outputs = [None] * self.solver_order
            self._lower_order_nums = 0
            self._last_sample = None
            self._this_order = 1
            return
        (
            self._step_index,
            model_outputs,
            self._lower_order_nums,
            self._last_sample,
            self._this_order,
        ) = state
        self._model_outputs = list(model_outputs)

    def _capture_stream_state(self) -> tuple:
        return (
            self._step_index,
            list(self._model_outputs),
            self._lower_order_nums,
            self._last_sample,
            self._this_order,
        )

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_tokens = self.set_timesteps(
            num_inference_steps,
            device=device,
            sigma_shift=shift_override,
        )
        return step_tokens.to(dtype=dtype), step_tokens

    @staticmethod
    def _alpha_sigma(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1.0 - sigma, sigma

    def _convert_model_output(
        self, model_output: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        sigma = self.sigmas[self._step_index].to(sample.device)
        return sample - sigma * model_output

    def _uni_p_update(self, sample: torch.Tensor, order: int) -> torch.Tensor:
        m0 = self._model_outputs[-1]
        if m0 is None:
            raise RuntimeError("UniPC predictor has no model output.")
        sigma_t = self.sigmas[self._step_index + 1].to(sample.device)
        sigma_s0 = self.sigmas[self._step_index].to(sample.device)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = torch.expm1(hh)
        b_h = torch.expm1(hh)

        differences = []
        rks = []
        for index in range(1, order):
            previous = self._model_outputs[-(index + 1)]
            if previous is None:
                raise RuntimeError("UniPC predictor history is incomplete.")
            sigma_si = self.sigmas[self._step_index - index].to(sample.device)
            alpha_si, sigma_si = self._alpha_sigma(sigma_si)
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            differences.append((previous - m0) / rk)

        result = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
        if differences:
            if order == 2:
                correction = 0.5 * differences[0]
            else:
                coefficients = self._solve_coefficients(
                    rks, h, sample.device, sample.dtype, predictor=True
                )
                correction = torch.einsum(
                    "k,bkc...->bc...", coefficients, torch.stack(differences, dim=1)
                )
            result = result - alpha_t * b_h * correction
        return result.to(sample.dtype)

    @staticmethod
    def _system(rks: list[torch.Tensor], h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full_rks = torch.stack([*rks, torch.ones_like(h)])
        hh = -h
        h_phi_k = torch.expm1(hh) / hh - 1
        b_h = torch.expm1(hh)
        factorial = 1
        rows, rhs = [], []
        for power in range(len(full_rks)):
            rows.append(full_rks.pow(power))
            rhs.append(h_phi_k * factorial / b_h)
            factorial *= power + 2
            h_phi_k = h_phi_k / hh - 1 / factorial
        return torch.stack(rows), torch.stack(rhs)

    def _solve_coefficients(
        self,
        rks: list[torch.Tensor],
        h: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        predictor: bool,
    ) -> torch.Tensor:
        matrix, rhs = self._system(rks, h)
        if predictor:
            matrix, rhs = matrix[:-1, :-1], rhs[:-1]
        return torch.linalg.solve(matrix.to(device), rhs.to(device)).to(dtype)

    def _uni_c_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        m0 = self._model_outputs[-1]
        if m0 is None:
            raise RuntimeError("UniPC corrector has no model output.")
        sigma_t = self.sigmas[self._step_index].to(this_sample.device)
        sigma_s0 = self.sigmas[self._step_index - 1].to(this_sample.device)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = torch.expm1(hh)
        b_h = torch.expm1(hh)

        differences = []
        rks = []
        for index in range(1, order):
            previous = self._model_outputs[-(index + 1)]
            if previous is None:
                raise RuntimeError("UniPC corrector history is incomplete.")
            sigma_si = self.sigmas[self._step_index - (index + 1)].to(this_sample.device)
            alpha_si, sigma_si = self._alpha_sigma(sigma_si)
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            differences.append((previous - m0) / rk)

        base = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        if order == 1:
            history_correction = 0
            current_coefficient = torch.tensor(0.5, device=this_sample.device, dtype=this_sample.dtype)
        else:
            coefficients = self._solve_coefficients(
                rks, h, this_sample.device, this_sample.dtype, predictor=False
            )
            history_correction = torch.einsum(
                "k,bkc...->bc...", coefficients[:-1], torch.stack(differences, dim=1)
            )
            current_coefficient = coefficients[-1]
        result = base - alpha_t * b_h * (
            history_correction + current_coefficient * (this_model_output - m0)
        )
        return result.to(this_sample.dtype)

    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        stream_id: str | None = None,
    ) -> torch.Tensor:
        if stream_id is not None:
            self._restore_stream_state(self._stream_states.get(str(stream_id)))
        if not self.timesteps.numel():
            raise RuntimeError("Call set_timesteps before step.")
        if self._step_index >= len(self.timesteps):
            raise IndexError("Scheduler has no remaining inference steps.")
        del timestep  # EasyWAM steps sequentially; official UniPC also tracks an internal index.

        converted = self._convert_model_output(model_output, sample)
        if self._step_index > 0 and self._last_sample is not None:
            sample = self._uni_c_update(
                converted, self._last_sample, sample, self._this_order
            )

        for index in range(self.solver_order - 1):
            self._model_outputs[index] = self._model_outputs[index + 1]
        self._model_outputs[-1] = converted

        available = self.solver_order
        if self.lower_order_final:
            available = min(available, len(self.timesteps) - self._step_index)
        self._this_order = min(available, self._lower_order_nums + 1)
        self._last_sample = sample
        previous = self._uni_p_update(sample, self._this_order)
        self._lower_order_nums = min(self.solver_order, self._lower_order_nums + 1)
        self._step_index += 1
        if stream_id is not None:
            self._stream_states[str(stream_id)] = self._capture_stream_state()
        return previous

    def sample_training_t(self, batch_size: int, device, dtype: torch.dtype) -> torch.Tensor:
        if self.time_distribution == "logitnormal":
            sigma = torch.randn(batch_size, device=device, dtype=torch.float32).sigmoid()
        else:
            sigma = torch.rand(batch_size, device=device, dtype=torch.float32)
        sigma = _shift_sigma(sigma, self.shift)
        return sigma.to(dtype)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = timestep.float().reshape(-1, *([1] * (clean.ndim - 1)))
        sigma = sigma.to(clean.dtype)
        return (1 - sigma) * clean + sigma * noise

    @staticmethod
    def training_target(clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - clean

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(timestep)
