"""
The abstraction layer for printer communication.
CURRENT ROLE: Returns mock status codes to test the software logic without a physical printer.
FUTURE ROLE: Will send actual HTTP/API requests to the printer's firmware (OctoPrint/Moonraker) to check status (is_printing) and execute commands (pause_print) upon defect detection.
"""