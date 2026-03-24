from typing import Optional, Union, Tuple
import requests

def get_input(
    label: str,
    input_type: str, 
    col,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    options: Optional[Tuple] = None
):
    if input_type.lower() == 'number':
        return col.number_input(
            label=label,
            min_value=min_val,
            max_value=max_val,
            help=(
                f"Minimum value is {min_val}" if max_val is None else
                f"Maximum value is {max_val}" if min_val is None else
                f"Value range from {min_val} - {max_val}"
            )
        )

    elif input_type.lower() == 'selectbox':
        return col.selectbox(
            label=label,
            options=options,
            index=None,
            placeholder="Select option"
        )

    else:
        raise ValueError("input_type must 'number' or 'selectbox'")
    
def check_api_health(API_URL:str):
    try:
        res = requests.get(f"{API_URL}/health", timeout=2)
        if res.status_code == 200:
            return True
        return False
    except:
        return False