import enum
from .utils import find_closest_index

class StateType(enum.Enum):
    GB_TRACK = 'GB_TRACK' 
    TRAILING = 'TRAILING' 
    OVERTAKE = 'OVERTAKE'

def string_to_state_type(state_string: str) -> StateType:
    """
    Converts a string to a StateType enum.
    """
    try:
        return StateType[state_string]
    except KeyError:
        return None
    
def state_type_to_string(state_type: StateType) -> str:
    """
    Converts a StateType enum to a string.
    """
    if isinstance(state_type, StateType):
        return state_type.value
    else:
        raise ValueError(f"Invalid state type: {state_type}")

def DefaultStateLogic(state_machine):
    """
    This is a global state that incorporates the other states
    """ 
    # Convert above to if statements
    if state_machine.state == StateType.GB_TRACK:
        return GlobalTracking(state_machine)
    elif state_machine.state == StateType.TRAILING:
        return Trailing(state_machine)
    elif state_machine.state == StateType.OVERTAKE:
        return Overtaking(state_machine)
    else:
        raise NotImplementedError(f"State {state_machine.state} not recognized")

"""
Here we define the behaviour in the different states.
Every function should be fairly concise, and output an array.
"""
def GlobalTracking(state_machine):
    curr_s = state_machine.car_s
    s_idx = find_closest_index(state_machine.wpnts_s_array, curr_s)
    return [state_machine.glb_wpnts[(s_idx + i)%state_machine.num_glb_wpnts] for i in range(state_machine.n_loc_wpnts)]

def Trailing(state_machine):
    # This allows us to trail on the last valid spline if necessary
    curr_s = state_machine.car_s
    s_idx = find_closest_index(state_machine.wpnts_s_array, curr_s)
    if state_machine.last_valid_avoidance_wpnts is not None:
        spline_wpts = state_machine.get_spline_wpts()
        return [spline_wpts[(s_idx + i)%state_machine.num_glb_wpnts] for i in range(state_machine.n_loc_wpnts)]
    else:
        return [state_machine.glb_wpnts[(s_idx + i)%state_machine.num_glb_wpnts] for i in range(state_machine.n_loc_wpnts)]

def Overtaking(state_machine):
    spline_wpts = state_machine.get_spline_wpts()
    s_idx = find_closest_index(state_machine.wpnts_s_array, state_machine.car_s)
    return [spline_wpts[(s_idx + i)%state_machine.num_glb_wpnts] for i in range(state_machine.n_loc_wpnts)]


# ------------------ State transitions ----------------------
def dummy_transition(state_machine)->str:
    return StateType.GB_TRACK
        
def timetrials_transition(state_machine)->str:
    return StateType.GB_TRACK

def head_to_head_transition(state_machine)->str:
    if state_machine.state == StateType.GB_TRACK:
        return SplineGlobalTrackingTransition(state_machine)
    elif state_machine.state == StateType.TRAILING:
        return SplineTrailingTransition(state_machine)
    elif state_machine.state == StateType.OVERTAKE:
        return SplineOvertakingTransition(state_machine)
    else:
        raise NotImplementedError(f"State {state_machine.state} not recognized")


def SplineGlobalTrackingTransition(state_machine) -> StateType:
    """Transitions for being in `StateType.GB_TRACK`"""
    if state_machine._check_gbfree:
        return StateType.GB_TRACK
    else:
        return StateType.TRAILING


def SplineTrailingTransition(state_machine) -> StateType:
    """Transitions for being in `StateType.TRAILING`"""
    gb_free = state_machine._check_gbfree
    ot_sector = state_machine._check_ot_sector

    if not gb_free and not ot_sector:
        return StateType.TRAILING
    elif gb_free and state_machine._check_close_to_raceline:
        return StateType.GB_TRACK
    elif (
        not gb_free
        and ot_sector
        and state_machine._check_availability_spline_wpts
        and state_machine._check_ofree
    ):
        return StateType.OVERTAKE
    else:
        return StateType.TRAILING


def SplineOvertakingTransition(state_machine) -> StateType:
    """Transitions for being in `StateType.OVERTAKE`"""
    in_ot_sector = state_machine._check_ot_sector
    spline_valid = state_machine._check_availability_spline_wpts
    o_free = state_machine._check_ofree

    # If spline is on an obstacle we trail
    if not o_free:
        return StateType.TRAILING
    if in_ot_sector and o_free and spline_valid:
        return StateType.OVERTAKE
    # If spline becomes unvalid while overtaking, we trail
    elif in_ot_sector and not spline_valid and not o_free:
        return StateType.TRAILING
    # go to GB_TRACK if not in ot_sector and the GB is free
    elif not in_ot_sector and state_machine._check_gbfree:
        return StateType.GB_TRACK
    # go to Trailing if not in ot_sector and the GB is not free
    else:
        return StateType.TRAILING