import sys
import traceback
from typing import Any
from src.logger import logging

def error_message_detail(error: Exception, error_detail: Any) -> str:
    """
    Built a detailed error message including filename and line number.

    """
    # exc_info returns (type, value, tb); tb is at index 2
    _, _, tb = error_detail.exc_info()
    # tb may be None (but here it's expected to be present)
    if tb is not None:
        # Walk to the last frame in the traceback (where exception occurred)
        last_tb = tb
        while last_tb.tb_next:
            last_tb = last_tb.tb_next
        file_name = last_tb.tb_frame.f_code.co_filename
        line_no = last_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_no = 0

    error_message = (
        f"Error occurred in Python script [{file_name}] "
        f"at line [{line_no}]. Error message: [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: Any):
        # Call base Exception init so exception chaining/printing works
        super().__init__(str(error_message))
        # Store our friendly detailed message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message

"""Basically we do this to get a detailed version of an error message at that perticlar
line and the possible cause for the error which helps in debugging huge codes in huge projects
especially when multiple people are working on it and could help in logging."""