import sys
from networksecurity.logging.logger import logging


class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys): 
        """
        error_message: str  
        error_details: object of sys module 
        Arguments:
            error_message: str
            error_details: object of sys module
        Returns:
            None
        """
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()     # exc_info() returns a tuple of three values: (type, value, traceback) available in the sys module. We are interested in the traceback object which is the third value in the tuple.

        self.line_number = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename 

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(self.file_name,self.line_number,self.error_message)

