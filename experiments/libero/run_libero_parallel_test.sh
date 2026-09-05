#!/bin/bash
export PATH="/root/miniforge3/envs/easywam/bin:$PATH"

run_libero_eval() {
    local task_list_file=$1
    echo "task_file: $task_list_file"

    require_non_empty() {
        local var_name="$1"
        local var_val="${!var_name}"
        if [ -z "$var_val" ]; then
            echo "Error: required variable $var_name is not set"
            exit 1
        fi
    }
    
    ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
    export ROOT_DIR
    RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
    export RUN_ID
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RUN_ID"}
    export OUTPUT_DIR  # Use run_id as the output subdirectory
    SESSION_NAME="libero_test_v3"
    EXP_NAME=${EXP_NAME:-""}
    export EXP_NAME

    echo "EXP_NAME: $EXP_NAME"
    
    mkdir -p "$OUTPUT_DIR"
    echo "Evaluation results will be saved to: $OUTPUT_DIR"

    cp "$task_list_file" "$OUTPUT_DIR/"
    task_list_file="$OUTPUT_DIR/$(basename $task_list_file)"
    echo "Task list file copied to: $task_list_file"
    
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        require_non_empty "NUM_GPUS"
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
    else
        AVAILABLE_GPUS=$CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    fi
    export NUM_GPUS
    echo "NUM_GPUS: $NUM_GPUS, AVAILABLE_GPUS: $AVAILABLE_GPUS"

    IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"

    require_non_empty "MAX_TASKS_PER_GPU"
    require_non_empty "NUM_TRIALS"
    TMUX_GRID_ROWS=${TMUX_GRID_ROWS:-1}
    TMUX_GRID_COLS=${TMUX_GRID_COLS:-$((MAX_TASKS_PER_GPU + 1))}
    GRID_ROWS=$TMUX_GRID_ROWS
    GRID_COLS=$TMUX_GRID_COLS
    MAX_PANES=$((GRID_ROWS * GRID_COLS - 1))
    if [ "$MAX_PANES" -le 0 ]; then
        echo "Error: invalid tmux grid configuration, TMUX_GRID_ROWS=$TMUX_GRID_ROWS TMUX_GRID_COLS=$TMUX_GRID_COLS"
        exit 1
    fi
    
    GPU_LOAD_FILE="$OUTPUT_DIR/gpu_load.txt"
    TASK_GPU_MAP_FILE="$OUTPUT_DIR/task_gpu_map.txt"
    TASK_STATUS_DIR="$OUTPUT_DIR/task_status"
    TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
    FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"

    mkdir -p "$TASK_STATUS_DIR" "$TASK_LOG_DIR"
    : > "$FAILED_TASKS_FILE"
    
    init_gpu_load_tracking() {
        > "$GPU_LOAD_FILE"
        > "$TASK_GPU_MAP_FILE"
        for gpu in "${GPU_ARRAY[@]}"; do
            echo "$gpu:0" >> "$GPU_LOAD_FILE"
        done
        echo "GPU load tracking initialized: $GPU_LOAD_FILE"
    }
    
    get_gpu_load() {
        local gpu_id=$1
        local load=$(grep "^$gpu_id:" "$GPU_LOAD_FILE" | cut -d: -f2)
        echo "${load:-0}"
    }
    
    update_gpu_load() {
        local gpu_id=$1
        local new_load=$2
        # Use a temporary file to keep the update atomic
        local temp_file="$GPU_LOAD_FILE.tmp"
        
        if [ -f "$GPU_LOAD_FILE" ]; then
            grep -v "^${gpu_id}:" "$GPU_LOAD_FILE" > "$temp_file" 2>/dev/null || true
        else
            > "$temp_file"
        fi
        
        echo "${gpu_id}:${new_load}" >> "$temp_file"
        mv "$temp_file" "$GPU_LOAD_FILE"
    }
    
    increment_gpu_load() {
        local gpu_id=$1
        local current_load=$(get_gpu_load $gpu_id)
        local new_load=$((current_load + 1))
        update_gpu_load $gpu_id $new_load
        echo $new_load
    }
    
    decrement_gpu_load() {
        local gpu_id=$1
        local current_load=$(get_gpu_load $gpu_id)
        local new_load=$((current_load - 1))
        [ $new_load -lt 0 ] && new_load=0
        update_gpu_load $gpu_id $new_load
        echo $new_load
    }
    
    find_least_loaded_gpu() {
        local min_load=999999
        local best_gpu=""
        for gpu in "${GPU_ARRAY[@]}"; do
            local load=$(get_gpu_load $gpu)
            if [ $load -lt $min_load ] && [ $load -lt $MAX_TASKS_PER_GPU ]; then
                min_load=$load
                best_gpu=$gpu
            fi
        done
        echo $best_gpu
    }
    
    show_gpu_status() {
        echo "=== GPU Load Status ==="
        for gpu in "${GPU_ARRAY[@]}"; do
            local load=$(get_gpu_load $gpu)
            local percentage=$((load * 100 / MAX_TASKS_PER_GPU))
            printf "GPU %s: %d/%d tasks (%d%%)\n" "$gpu" "$load" "$MAX_TASKS_PER_GPU" "$percentage"
        done
        echo "=================="
    }
    
    show_debug_info() {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Debug Info ==="
        
        if [ -f "$GPU_LOAD_FILE" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU load file contents:"
            cat "$GPU_LOAD_FILE" | while IFS=: read gpu load; do
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   GPU$gpu: $load"
            done
        fi
        
        if [ -f "$TASK_GPU_MAP_FILE" ]; then
            local map_count=$(wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of running tasks: $map_count"
            if [ $map_count -gt 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running tasks:"
                cat "$TASK_GPU_MAP_FILE" | while IFS=: read task_info gpu_id; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   $task_info -> GPU$gpu_id"
                done
            fi
        fi
        
        if [ -f "$PENDING_TASKS_FILE" ]; then
            local pending_count=$(wc -l < "$PENDING_TASKS_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of pending tasks: $pending_count"
        fi

        if [ -f "$FAILED_TASKS_FILE" ]; then
            local failed_count=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of failed tasks: $failed_count"
        fi
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ==================="
    }
    
    record_task_gpu_mapping() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        echo "$suite,$task_id:$gpu_id" >> "$TASK_GPU_MAP_FILE"
    }
    
    get_task_gpu() {
        local suite=$1
        local task_id=$2
        local mapping=$(grep "^$suite,$task_id:" "$TASK_GPU_MAP_FILE" | cut -d: -f2)
        echo "${mapping:-}"
    }
    
    remove_task_gpu_mapping() {
        local suite=$1
        local task_id=$2
        local temp_file="$TASK_GPU_MAP_FILE.tmp"
        grep -v "^$suite,$task_id:" "$TASK_GPU_MAP_FILE" > "$temp_file" 2>/dev/null || true
        mv "$temp_file" "$TASK_GPU_MAP_FILE"
    }

    append_unique_pending_task() {
        local suite=$1
        local task_id=$2
        local task_key="$suite,$task_id"
        if [ ! -f "$PENDING_TASKS_FILE" ] || ! grep -qxF "$task_key" "$PENDING_TASKS_FILE"; then
            echo "$task_key" >> "$PENDING_TASKS_FILE"
        fi
    }

    mark_task_failed() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local return_code=$4
        local log_file=$5
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$timestamp,$suite,$task_id,gpu=$gpu_id,rc=$return_code,log=$log_file" >> "$FAILED_TASKS_FILE"
    }
    
    CKPT=${CKPT:-""}
    export CKPT
    CONFIG=${CONFIG:-""}
    require_non_empty "CKPT"
    require_non_empty "CONFIG"
    CONFIG="${CONFIG#configs/}" # delete prefix configs/
    CONFIG="${CONFIG#task/}" # delete prefix task/
    CONFIG="${CONFIG%.yaml}" # delete suffix .yaml
    export CONFIG

    echo "CKPT: $CKPT"
    echo "CONFIG: $CONFIG"
    echo "ROOT_DIR: $ROOT_DIR"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "MAX_TASKS_PER_GPU: $MAX_TASKS_PER_GPU"
    
    init_gpu_load_tracking

    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        tmux kill-session -t $SESSION_NAME
        echo "Session '$SESSION_NAME' has been deleted"
    fi

    tmux new-session -d -s $SESSION_NAME

    create_grid_layout() {
        local window=$1
        if [ $window -gt 0 ]; then
            if ! tmux list-windows -t $SESSION_NAME | grep -q "^$window:"; then
                tmux new-window -t $SESSION_NAME:$window
            fi
        fi
        
        local pane_count=$(tmux list-panes -t $SESSION_NAME:$window | wc -l)
        
        for ((i=pane_count; i<GRID_ROWS*GRID_COLS-1; i++)); do
            tmux split-window -t $SESSION_NAME:$window
            tmux select-layout -t $SESSION_NAME:$window tiled
        done
    }
    
    create_grid_layout 0
    
    NEXT_PANE_INDEX=0
    
    ensure_pane_exists() {
        local window_id=$1
        local pane_id=$2
        
        if [ $window_id -gt 0 ]; then
            if ! tmux list-windows -t $SESSION_NAME | grep -q "^$window_id:" 2>/dev/null; then
                tmux new-window -t $SESSION_NAME:$window_id 2>/dev/null
                create_grid_layout $window_id
            fi
        fi
        
        if [ $pane_id -eq 0 ] && [ $window_id -gt 0 ]; then
            create_grid_layout $window_id
        fi
    }
    
    launch_task_on_pane() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local pane_info=$4
        local status_file="$TASK_STATUS_DIR/${suite}_task${task_id}.status"
        local result_file="$OUTPUT_DIR/$suite/gpu${gpu_id}_task${task_id}_results.json"
        local log_file="$TASK_LOG_DIR/${suite}_task${task_id}_gpu${gpu_id}.log"
        
        rm -f "$status_file"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching task: $suite task_id=$task_id on GPU$gpu_id pane $pane_info"
        
        # When the task exits, write a status file so the scheduler can detect failures promptly.
        tmux select-pane -t $SESSION_NAME:$pane_info 2>/dev/null
        tmux send-keys -t $SESSION_NAME:$pane_info "clear" C-m 2>/dev/null
        tmux send-keys -t $SESSION_NAME:$pane_info "source ~/.bashrc && cd $ROOT_DIR && export EXP_NAME=$EXP_NAME && \
            STATUS_FILE='$status_file' LOG_FILE='$log_file' RESULT_FILE='$result_file' && \
            CUDA_VISIBLE_DEVICES=$gpu_id /root/miniforge3/envs/easywam/bin/python experiments/libero/eval_libero_single.py \
            task=$CONFIG ckpt=$CKPT \
            EVALUATION.task_suite_name=$suite EVALUATION.task_id=$task_id gpu_id=$gpu_id \
            EVALUATION.num_trials=$NUM_TRIALS EVALUATION.output_dir=$OUTPUT_DIR $EXTRA_ARGS > \"\$LOG_FILE\" 2>&1; \
            rc=\$?; \
            if [ \$rc -eq 0 ] && [ -f \"\$RESULT_FILE\" ]; then \
                echo \"SUCCESS|$gpu_id|\$rc|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; \
            else \
                echo \"FAILED|$gpu_id|\$rc|\$(date +%s)|\$LOG_FILE\" > \"\$STATUS_FILE\"; \
            fi" C-m 2>/dev/null
        return 0
    }

    launch_task() {
        local suite=$1
        local task_id=$2
        local gpu_id=$3
        local pane_info=$4

        record_task_gpu_mapping "$suite" "$task_id" "$gpu_id"
        local new_load=$(increment_gpu_load "$gpu_id")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Assigned task: $suite task_id=$task_id -> GPU$gpu_id (load: $new_load/$MAX_TASKS_PER_GPU)"
        launch_task_on_pane "$suite" "$task_id" "$gpu_id" "$pane_info"
    }
    
    cleanup_completed_tasks() {
        CLEANED_COUNT=0
        NEW_FAILURE_COUNT=0

        if [ ! -f "$TASK_GPU_MAP_FILE" ] || [ ! -s "$TASK_GPU_MAP_FILE" ]; then
            return 0
        fi

        local temp_map="$TASK_GPU_MAP_FILE.cleanup"
        > "$temp_map"

        while IFS=: read -r task_info gpu_id; do
            [ -z "$task_info" ] && continue
            local suite=$(echo "$task_info" | cut -d, -f1)
            local task_id=$(echo "$task_info" | cut -d, -f2)
            [ -z "$suite" ] || [ -z "$task_id" ] && continue

            local status_file="$TASK_STATUS_DIR/${suite}_task${task_id}.status"
            local any_result_pattern="$OUTPUT_DIR/$suite/gpu*_task${task_id}_results.json"

            if ls $any_result_pattern 1> /dev/null 2>&1; then
                local new_load=$(decrement_gpu_load "$gpu_id")
                rm -f "$status_file"
                ((CLEANED_COUNT++))
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task completed: $suite task_id=$task_id GPU$gpu_id released (load: $new_load/$MAX_TASKS_PER_GPU)"
                continue
            fi

            if [ -f "$status_file" ]; then
                IFS='|' read -r status status_gpu status_rc status_ts status_log < "$status_file"
                if [ "$status" = "FAILED" ]; then
                    local new_load=$(decrement_gpu_load "$gpu_id")
                    mark_task_failed "$suite" "$task_id" "$gpu_id" "${status_rc:-unknown}" "${status_log:-unknown}"
                    ((NEW_FAILURE_COUNT++))
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task failed: $suite task_id=$task_id rc=$status_rc GPU$gpu_id (current load: $new_load/$MAX_TASKS_PER_GPU)"
                    rm -f "$status_file"
                    continue
                fi
                if [ "$status" = "SUCCESS" ]; then
                    local new_load=$(decrement_gpu_load "$gpu_id")
                    rm -f "$status_file"
                    ((CLEANED_COUNT++))
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Task completed (status file): $suite task_id=$task_id GPU$gpu_id released (load: $new_load/$MAX_TASKS_PER_GPU)"
                    continue
                fi
            fi

            echo "$task_info:$gpu_id" >> "$temp_map"
        done < "$TASK_GPU_MAP_FILE"

        mv "$temp_map" "$TASK_GPU_MAP_FILE"
        return 0
    }

    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dynamic task scheduling..."
    
    PENDING_TASKS_FILE="$OUTPUT_DIR/pending_tasks.txt"
    cp "$task_list_file" "$PENDING_TASKS_FILE"
    
    local total_tasks=$(wc -l < "$task_list_file")
    local monitoring_interval=${MONITORING_INTERVAL:-10}  # Monitoring interval in seconds
    local last_status_time=0
    local status_interval=${STATUS_INTERVAL:-30}  # Status display interval in seconds
    local max_launch_per_round=${MAX_LAUNCH_PER_ROUND:-$MAX_TASKS_PER_GPU}
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tasks: $total_tasks"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Max tasks per GPU: $MAX_TASKS_PER_GPU"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Available GPUs: ${GPU_ARRAY[*]}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting the initial launch phase..."
    local initial_launched=0
    local max_initial_tasks=$((NUM_GPUS * MAX_TASKS_PER_GPU))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Planning to launch up to $max_initial_tasks initial tasks"
    
    local task_array=()
    
    while IFS=, read -r suite task_id; do
        [ -z "$suite" ] && continue
        task_array+=("$suite,$task_id")
    done < "$task_list_file"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded ${#task_array[@]} tasks"
    
    for task_info in "${task_array[@]}"; do
        [ $initial_launched -ge $max_initial_tasks ] && break
        
        suite=$(echo $task_info | cut -d, -f1)
        task_id=$(echo $task_info | cut -d, -f2)
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing task: suite=$suite, task_id=$task_id"
        
        gpu_id=$(find_least_loaded_gpu)
        if [ -z "$gpu_id" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] All GPUs are fully loaded, stopping initial launch"
            break
        fi
        
        # Compute pane assignment directly in the parent process
        window_id=$((NEXT_PANE_INDEX / MAX_PANES))
        pane_id=$((NEXT_PANE_INDEX % MAX_PANES))
        pane_info="$window_id.$pane_id"
        
        ensure_pane_exists $window_id $pane_id
        
        NEXT_PANE_INDEX=$((NEXT_PANE_INDEX + 1))
        
        launch_task "$suite" "$task_id" "$gpu_id" "$pane_info"
        
        ((initial_launched++))
        
        grep -v "^$suite,$task_id$" "$PENDING_TASKS_FILE" > "$PENDING_TASKS_FILE.tmp" || true
        mv "$PENDING_TASKS_FILE.tmp" "$PENDING_TASKS_FILE"
        
        sleep 0.5
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initial launch completed, started $initial_launched tasks"
    
    while true; do
        current_time=$(date +%s)

        cleanup_completed_tasks
        cleaned=$CLEANED_COUNT
        new_failures=$NEW_FAILURE_COUNT
        total_failed=$(wc -l < "$FAILED_TASKS_FILE" 2>/dev/null || echo 0)

        if [ "$new_failures" -gt 0 ]; then
            echo "Detected failed subtasks, stopping the scheduler. Failure details: $FAILED_TASKS_FILE"
            cat "$FAILED_TASKS_FILE"
            return 2
        fi

        total_completed=$(find "$OUTPUT_DIR" -type f -name "gpu*_task*_results.json" | wc -l)
        if [ "$total_completed" -eq "$total_tasks" ]; then
            echo "All tasks are complete!"
            break
        fi

        launched_this_round=0

        temp_pending="$PENDING_TASKS_FILE.processing"
        cp "$PENDING_TASKS_FILE" "$temp_pending" 2>/dev/null || continue

        > "$PENDING_TASKS_FILE"

        while IFS=, read -r suite task_id; do
            [ -z "$suite" ] && continue

            result_file_pattern="$OUTPUT_DIR/$suite/gpu*_task${task_id}_results.json"
            if ls $result_file_pattern 1> /dev/null 2>&1; then
                continue
            fi

            # The pending file should only keep tasks that are not running.
            running_gpu=$(get_task_gpu "$suite" "$task_id")
            if [ -n "$running_gpu" ]; then
                continue
            fi

            gpu_id=$(find_least_loaded_gpu)
            if [ -n "$gpu_id" ]; then
                window_id=$((NEXT_PANE_INDEX / MAX_PANES))
                pane_id=$((NEXT_PANE_INDEX % MAX_PANES))
                pane_info="$window_id.$pane_id"

                ensure_pane_exists "$window_id" "$pane_id"
                NEXT_PANE_INDEX=$((NEXT_PANE_INDEX + 1))

                launch_task "$suite" "$task_id" "$gpu_id" "$pane_info"
                ((launched_this_round++))

                if [ $launched_this_round -ge $max_launch_per_round ]; then
                    while IFS=, read -r remaining_suite remaining_task_id; do
                        [ -n "$remaining_suite" ] && append_unique_pending_task "$remaining_suite" "$remaining_task_id"
                    done
                    break
                fi
            else
                append_unique_pending_task "$suite" "$task_id"
            fi
        done < "$temp_pending"

        rm -f "$temp_pending"

        running_count=$(wc -l < "$TASK_GPU_MAP_FILE" 2>/dev/null || echo 0)
        pending_count=$(wc -l < "$PENDING_TASKS_FILE" 2>/dev/null || echo 0)

        if [ "$running_count" -eq 0 ] && [ "$pending_count" -eq 0 ] && [ "$total_completed" -lt "$total_tasks" ]; then
            echo "Scheduling inconsistency: no running tasks and no pending tasks, but not all tasks are complete."
            echo "Completed: $total_completed/$total_tasks, failed: $total_failed"
            [ -s "$FAILED_TASKS_FILE" ] && cat "$FAILED_TASKS_FILE"
            return 2
        fi
        
        if [ $((current_time - last_status_time)) -ge $status_interval ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Scheduling Status $(date '+%H:%M:%S') ==="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total tasks: $total_tasks"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $total_completed"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remaining: $((total_tasks - total_completed))"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $running_count"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pending: $pending_count"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: $total_failed"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launched this round: $launched_this_round"
            if [ "$cleaned" -gt 0 ] 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaned this round: $cleaned"
            fi
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] === GPU Load Status ==="
            for gpu in "${GPU_ARRAY[@]}"; do
                load=$(get_gpu_load $gpu)
                percentage=$((load * 100 / MAX_TASKS_PER_GPU))
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: $load/$MAX_TASKS_PER_GPU tasks ($percentage%)"
            done
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================="
            
            show_debug_info
            echo ""
            last_status_time=$current_time
        fi
        
        sleep $monitoring_interval
    done
    
    rm -f "$PENDING_TASKS_FILE" "$PENDING_TASKS_FILE.processing"

    echo "All tasks completed successfully!"
    echo "Generating evaluation report..."
    /root/miniforge3/envs/easywam/bin/python experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
}


if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then
        echo "Error: task file path is required"
        echo "Usage: $0 <task_file>"
        exit 1
    fi
    test_file="$1"
    run_libero_eval "$test_file"
    exit $?
fi
