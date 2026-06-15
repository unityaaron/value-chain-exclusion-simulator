# =====================================================================
# THE OBJECTIVE: This script is an automated conveyor belt. 
# It takes a raw, messy Excel file from the farm, cleans it, 
# feeds it to our frozen machine learning model, and saves the answers.
# =====================================================================

import pandas as pd
import joblib

# Step 2: Define our custom automation machine
# We create a function named 'run_automated_factory'. 
# It has an input slot called 'excel_path' which holds the name of our Excel file.

def run_automated_factory(excel_path):
    print('Conveyor Belt Started')

    # -----------------------------------------------------------------
    # PIPELINE STAGE 1: LOADING THE RAW DATA
    # -----------------------------------------------------------------
    # We use pandas to open the Excel file. header=1 means skip the very 
    # first row because our real column titles start on row 2.
    # The loaded table is stored in a temporary variable called 'raw_data'.

    raw_data=pd.read_excel(excel_path, header=1)
    print("📋 Real Column Names in Excel are:", raw_data.columns.tolist())
    print(f'Successfully loaded {len(raw_data)} farmers from the file')

    # -----------------------------------------------------------------
    # PIPELINE STAGE 2: CATEGORY CLEANING (ENCODING)
    # -----------------------------------------------------------------
    # Our machine learning model only understands numbers, not words. 
    # Here, we look at the 'Training Attended' column which has 'Yes' or 'No'.
    # We use .map() to automatically change every 'Yes' to 1 and every 'No' to 0.
    # We store these new numbers in a brand new column called 'Training_Encoded'.

    raw_data['Training_Encoded'] = raw_data['Training Attended'].map({'Yes':1, 'No': 0})
    raw_data['Fertilizer_x_Training'] = raw_data['Fertilizer Used (kg)'] * raw_data['Training_Encoded']
    print("✨ Data Cleaning and Feature Engineering completed automatically.")

    # -----------------------------------------------------------------
    # PIPELINE STAGE 4: PREPARING THE INPUT MATRIX (X)
    # -----------------------------------------------------------------
    # Our frozen model is very strict. It only knows how to predict if we give it 
    # the exact same 4 clues it learned during training, in the exact same order.
    # We select those 4 columns here and group them into a sub-table called 'clues_conveyor'.

    clues_conveyor = raw_data[['Farm Size (ha)', 'Fertilizer Used (kg)', 'Training_Encoded', 'Fertilizer_x_Training']]


    # -----------------------------------------------------------------
    # PIPELINE STAGE 5: LOADING THE TRAINED MODEL (THE BRAIN)
    # -----------------------------------------------------------------
    # We use joblib.load() to go into your 'machine_learning_model' folder, 
    # find the frozen model file 'jpil_random_forest_model.pkl', and load its 
    # mathematical brains into a live python variable called 'frozen_robot'.

    frozen_robot = joblib.load('machine_learning_model/jpil_random_forest_model.pkl')
    print("🧠 Frozen Random Forest model successfully awoken and loaded.")

    # -----------------------------------------------------------------
    # PIPELINE STAGE 6: MAKING PREDICTIONS
    # -----------------------------------------------------------------
    # We feed our prepared 'clues_conveyor' table into the live 'frozen_robot' model.
    # The robot runs its internal mathematical logic and outputs an array of numbers.
    # These numbers are the predicted yield change percentages for every single farmer.

    automated_predictions = frozen_robot.predict(clues_conveyor)


    # -----------------------------------------------------------------
    # PIPELINE STAGE 7: INTEGRATING THE RESULTS
    # -----------------------------------------------------------------
    # We take the list of predictions the robot gave us and glue them back onto 
    # our original data table as a brand new column named 'Predicted_Yield_Change_%'.

    raw_data['Predicted_Yield_Change %'] = automated_predictions
    print("✅ Pipeline complete! Showing the first 5 automated predictions:\n")

    # .head() prints out just the first 5 rows of our table so we can inspect it on screen.
    print(raw_data[['Training Attended', 'Predicted_Yield_Change %']].head())


    # -----------------------------------------------------------------
    # PIPELINE STAGE 8: EXPORTING THE FINAL DELIVERABLE
    # -----------------------------------------------------------------
    # Finally, we save the entire updated table (with our predictions included) 
    # back into a brand new Excel file called 'final_automated_predictions.xlsx'.
    # index=False ensures we don't save extra messy row index numbers into Excel.
    
    raw_data.to_excel('final_automated_predictions.xlsx', index=False)
    print('\n 💾 Saved final spreadsheet as: final_automated_predictions.xlsx')


# =====================================================================
# ACTIVATE THE SYSTEM
# =====================================================================
# Everything above is just building the machine. It won't run until we call it.
# Here, we turn the machine on by calling its name and dropping our current 
# spreadsheet into the input slot.

run_automated_factory('jpil_farmers_dataset.xlsx')




