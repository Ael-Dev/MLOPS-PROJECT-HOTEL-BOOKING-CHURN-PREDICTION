import sys
import traceback

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def _get_detailed_error_message(error_message: str, error_detail: sys) -> str:
        """
        Genera un mensaje de error detallado con el archivo y la línea donde ocurrió la excepción.
        """
        # Obtiene la información de la excepción
        exc_type, exc_obj, exc_tb = error_detail.exc_info()
        if exc_tb is None:
            return error_message  # Si no hay traceback, devuelve el mensaje original

        # Extrae el nombre del archivo y el número de línea
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        # Construye el mensaje detallado
        return f"Error in {file_name}, line {line_number}: {error_message}"

    def __str__(self):
        return self.error_message

