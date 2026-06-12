import sys          
import subprocess

from config import Config

import streamlit as st
from streamlit_autorefresh import st_autorefresh

#run this using:
# cd "c:\Users\Pasca\stuff\code\python\BEP\Fizzy code"
# streamlit run "BEP code/Python_State_Machine/dashboard.py"


# -----------------------------------------------------------------------------
# Automatically refresh the page
# -----------------------------------------------------------------------------

# Refresh the page every 1000 ms (1 second).
# This allows the dashboard to continuously read the latest values written by
# the robot controller.
st_autorefresh(interval=1000, key="dashboard_refresh")



# -----------------------------------------------------------------------------
# Page layout
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Fizzy Dashboard", page_icon= "Fizzy-oicon.png", layout="wide")

# Reduce the empty top margin on the page.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3rem; 
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Read the latest JSON values
# -----------------------------------------------------------------------------
config = Config()
config.load()


exercise_options = {
    "Select": "standby",
    "Opgepakt!": "walk",
    "In Balans": "balance",
    "Let OP!": "table"
}

# Create reverse mapping: standby -> Select, walk -> Loop mee, ...
reverse_exercise_options = {
    value: key for key, value in exercise_options.items()
}

# Initialize session state once
if "exercise_select" not in st.session_state:
    st.session_state.exercise_select = reverse_exercise_options.get(
        config["program"],
        "Select"
    )

if "difficulty_slider" not in st.session_state:
    st.session_state.difficulty_slider = int(config["difficulty"])

if "sensitivity_slider" not in st.session_state:
    st.session_state.sensitivity_slider = int(config["sensitivity"])

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

logo_col, main_col = st.columns([1, 3])

# with logo_col:
#     st.image("Fizzy-o.svg", width=150)

with main_col:
    #st.title("Welkom!")
    st.subheader("Settings")



  
    # -------------------------------
    # Power button logic starting and stopping the main.py file
    # -------------------------------
    if 'robot_process' not in st.session_state:
        st.session_state.robot_process = None
    
    # -----------------------------
    # Determine status
    # -----------------------------

    # Default status: OFF
    status_icon = "⚫ OFF"

    # Check if main.py is running
    if st.session_state.robot_process is not None:
        if st.session_state.robot_process.poll() is None:
            # Process alive
            status_icon = "🟢 ON"
        else:
            # Process died
            st.session_state.robot_process = None
            status_icon = "⚫ OFF"
    # else:
    #     # Process not started
    #     if config.get("power", False):
    #         # JSON says power is True but process not running → fix JSON
    #         config["power"] = False
    #         config.save_settings()

    # -----------------------------
    # Display button and status
    # -----------------------------
    
    # The button toggles the power state
    if st.button("POWER"):
        if status_icon.startswith("⚫"):
            # Currently OFF → turn ON
            config["power"] = True
            config.save_settings()
            if st.session_state.robot_process is None:
                st.session_state.robot_process = subprocess.Popen(
                    [sys.executable, "BEP code/Python_State_Machine/main.py"]
                )
        else:
            # Currently ON → turn OFF
            config["power"] = False
            config.save_settings()
            # main.py will detect power=False and exit cleanly

    st.markdown(f"**Status:** {status_icon}")
    
   
    # -------------------------------------------------------------------------
    # Exercise selection
    # -------------------------------------------------------------------------

    selected = st.selectbox(
        "Exercise",
        list(exercise_options.keys()),
        key="exercise_select"
    )

    # -------------------------------------------------------------------------
    # Difficulty slider
    # -------------------------------------------------------------------------

    difficulty = st.slider(
        "Difficulty",
        min_value=1,
        max_value=3,
        key="difficulty_slider"
    )

    # sensitivity = st.slider(
    #     "Sensitivity",
    #     min_value=1.0,
    #     max_value=5.0,
    #     step=0.1,
    #     key="sensitivity_slider"
    # )


    # -----------------------------------------------------------------------------
    # Live status display
    # -----------------------------------------------------------------------------

    # Convert time in seconds into mm:ss format.
    time_seconds = int(config.get("time_seconds", 0))
    minutes = time_seconds // 60
    seconds = time_seconds % 60
    formatted_time = f"{minutes:02d}:{seconds:02d}"

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        # Reads live value from JSON file.
        st.metric("Number of Drops", config["taps"])

    with metric_col2:
        # Reads live time from JSON file.
        st.metric("Time", formatted_time)

    #with metric_col3:
       # st.metric("Current Program", config["program"])


    # -----------------------------------------------------------------------------
    # Save settings button
    # -----------------------------------------------------------------------------

    # Only the settings from the dashboard are written when this button is pressed.
    # The robot controller can continue updating taps/time in the same file.
    if st.button("Apply Settings"):
        config["program"] = exercise_options[st.session_state.exercise_select]
        config["difficulty"] = st.session_state.difficulty_slider
        config["sensitivity"] = st.session_state.sensitivity_slider
        config.save_settings()

        st.success("Settings saved!")
    

    # # -----------------------------------------------------------------------------
    # # Debug section
    # # -----------------------------------------------------------------------------

    # with st.expander("Show current JSON file: settings"):
    #     st.json(config.settings_dict())
        
    # with st.expander("Show current JSON file: runtime"):
    #     st.json(config.runtime_dict())