import sys ##allows access to Python system specific details (like exceptions, traceback, etc.).
from src.Logger import logging ##imports your custom logger so you can log errors.

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

#def error_message_detail(...) - function that builds a detailed error message.
#error - the actual error/exception (ValueError, TypeError, etc.).
#error_detail:sys - type hint → says this function expects system error info.
#error_detail.exc_info() - returns (exception_type, exception_value, traceback_object).
#_, _, exc_tb = ... - ignores type and value, keeps only traceback_object (which has file & line info).
#exc_tb.tb_frame - gets the frame (context) of the error.
#.f_code.co_filename - gives you the file name where the error happened.
    

class CustomException(Exception): ##(Exception) - inherits from Python’s built-in Exception.
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message) ##super().__init__(error_message) - call parent Exception class so it behaves like a normal error.
        self.error_message = error_message_detail(error_message,error_detail = error_detail)

    def __str__(self):
        return self.error_message ##Whenever you print(CustomException), it shows the formatted error string instead of a random object.
    