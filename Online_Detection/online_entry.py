from online_detection_v3 import OnlineIdleDetector
from sftp_get import get_file
from send_email import send_email
import pandas as pd
from event_logging import EventLogger
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any

prev_file = None
end_ts = None
state = None

@ dataclass
class EmailState:
    idle_start: Any | None = None
    currentTime: Any | None = None
    idle_length: pd.Timedelta | None = None
    idle_end: bool = False
    notified: bool = False
    warning_suppressed: bool = False
    warning_sent: bool = False
    in_idle: bool = False


    def reset(self):
        for f in fields(self):
            setattr(self, f.name, f.default)


def write_log(msg):
    msg = str(msg)
    with open('var_log.txt', 'a', encoding="utf-8") as f:
        print(msg)
        f.write(msg + '\n')

def email_sender(email_var):
    # idle_span_len = email_var.idle_length.total_seconds() / 60
    idle_span_len = email_var.idle_length
    if idle_span_len > 20 and not email_var.notified:
        content = f"Idle time of {idle_span_len} minutes detected. Please check."
        condition = "Idle First Detected"
        # msg = [condition, content]
        print(content)
        # send_email(condition, content)
        email_var.notified = True
    elif idle_span_len >= 45 and not email_var.warning_sent:
        content = f"Idle time of {idle_span_len} minutes detected. Please check."
        condition = "Idle First Warning"
        print(content)
        # send_email(condition, content)
        email_var.warning_sent = True
    elif idle_span_len >= 75 and not email_var.warning_suppressed:
        content = ("Idle time of {} minutes detected. Please check. \nAttention: This is "
                   "the last reminder, no future warning will be sent.".format(idle_span_len))
        condition = "Idle Last Warning"
        # msg = [condition, content]
        # send_email(condition, content)
        print(content)
        # pass
        email_var.warning_suppressed = True
    else:
        return


def notify_lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None):
    msg = (f"[LOWVAR-START] id={run_id} start={start_idx} first_valid={first_valid_idx} "
          f"(min_len={min_len}, {threshold}) ctx={context}")
    write_log(msg)

def notify_lowvar_end(run_id, start_idx, end_idx, length, context=None):
    msg = f"[LOWVAR-END]   id={run_id} start={start_idx} end={end_idx} len={length} ctx={context}"
    write_log(msg)

def notify_off_start(start_idx, first_valid_idx, min_len, context=None):
    msg = f"[OFF-START]    start={start_idx} first_valid={first_valid_idx} (min_len={min_len}) ctx={context}"
    write_log(msg)

def notify_off_end(start_idx, end_idx, length, context=None):
    msg = f"[OFF-END]      start={start_idx} end={end_idx} len={length} ctx={context}"
    write_log(msg)

def make_callbacks(batch_series, det, run_id_str, source_file, evt_logger, email_var):
    det_params = dict(
        mad_win_s=int(det.mad_win),     # if your window is in samples (1/min), convert to seconds
        mad_threshold_kw=det.mad_threshold,
        off_min_len_s=int(det.off_min_len),
        low_var_min_len_s=int(det.min_len)
    )

    def lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None):
        evt_logger.on_lowvar_start(
            run_id=run_id,
            start_idx=start_idx,
            first_valid_idx=first_valid_idx,
            min_len=min_len,
            threshold=threshold,
            batch_series=batch_series,
            det_params=det_params,
            run_id_str=run_id_str,
            source_file=source_file
        )
        notify_lowvar_start(run_id, start_idx, first_valid_idx, min_len, threshold, context=None)

        # start = datetime.fromisoformat(start_idx)
        # current_end = datetime.fromisoformat(first_valid_idx)
        # current_length = first_valid_idx - start_idx
        # email_var.idle_start = start_idx
        # email_var.idle_length = current_length
        # email_var.currentTime = first_valid_idx
        email_var.in_idle = True


    def lowvar_end(run_id, start_idx, end_idx, length, context=None):
        evt_logger.on_lowvar_end(
            run_id=run_id,
            start_idx=start_idx,
            end_idx=end_idx,
            length=length,
            batch_series=batch_series
        )
        notify_lowvar_end(run_id, start_idx, end_idx, length, context=None)
        email_var.idle_end = True
        # email_var.reset()

    return lowvar_start, lowvar_end


evt_logger = EventLogger(
            log_path=Path("event_log.jsonl"),
            machine_id="Machine201_motor_sum")

email_state = EmailState()

for i in range(2448  , 1, -1):
    current_info = get_file(prev_file, end_ts, i)
    prev_file, end_ts, function_call, motor_sum, file = current_info
    # print('END_TS:', end_ts, type(end_ts))

    if function_call:
        print('Data Received, Starting Detection')

        det = OnlineIdleDetector(
            mad_win=6, hampel_win=4, n_sigmas=3.0,
            mad_threshold=5, min_len=15,
            off_level=1.0, off_min_len=10,
            tiny_threshold=5, ema_alpha=0.25,
            on_off_start=notify_off_start,
            on_off_end=notify_off_end,
            context={"device": "Motor Sum"},
        )

        batch_start = motor_sum.index.min().strftime("%Y-%m-%dT%H:%M:%S%z") if (
            motor_sum.index.tzinfo) else motor_sum.index.min().strftime("%Y-%m-%dT%H:%M:%S")
        run_id_str = f"{batch_start}__MOTOR1"
        source_file = file

        cb_lowvar_start, cb_lowvar_end = make_callbacks(motor_sum, det, run_id_str, source_file,
                                                        evt_logger, email_state)
        det.on_lowvar_start = cb_lowvar_start
        det.on_lowvar_end = cb_lowvar_end

        # if email_state.in_idle and email_state.idle_start is not None:
        #     email_state.idle_length = end_ts - email_state.idle_start
        #     email_state.currentTime = end_ts
        #     email_sender(email_state)

        out = det.process_batch(motor_sum, state=state)

        state = out["state"]
        # print(email_state.in_idle, out['state'].email_len)
        if out['state'].email_len >= 15 and email_state.in_idle:
            email_state.idle_length = out['state'].email_len
            # print(out['state'].lowvar_len_so_far, out['state'].email_len)
            email_sender(email_state)
            if email_state.idle_end:
                email_state.reset()
                state.email_len = 0


