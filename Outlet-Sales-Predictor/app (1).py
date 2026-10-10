from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# LOAD TRAINED KNN MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "knn_regression_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    print("KNN model loaded successfully.")
    print("Model path:", MODEL_PATH)

except Exception as e:
    print("ERROR loading KNN model:")
    print(e)
    model = None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:
            return render_template(
                "index.html",
                error="KNN model could not be loaded."
            )


        # ----------------------------------------------------
        # Get values from HTML form
        # ----------------------------------------------------

        item_weight = float(
            request.form["item_weight"]
        )

        item_fat_content = int(
            request.form["item_fat_content"]
        )

        item_visibility = float(
            request.form["item_visibility"]
        )

        item_mrp = float(
            request.form["item_mrp"]
        )

        outlet_size = int(
            request.form["outlet_size"]
        )

        outlet_location_type = int(
            request.form["outlet_location_type"]
        )

        outlet_establishment_year = int(
            request.form["outlet_establishment_year"]
        )

        item_type = request.form["item_type"]

        outlet_identifier = request.form[
            "outlet_identifier"
        ]

        outlet_type = request.form[
            "outlet_type"
        ]


        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame({

            "Item_Weight": [
                item_weight
            ],

            "Item_Fat_Content": [
                item_fat_content
            ],

            "Item_Visibility": [
                item_visibility
            ],

            "Item_MRP": [
                item_mrp
            ],

            "Outlet_Establishment_Year": [
                outlet_establishment_year
            ],

            "Outlet_Size": [
                outlet_size
            ],

            "Outlet_Location_Type": [
                outlet_location_type
            ]
        })


        # ====================================================
        # ITEM TYPE ONE-HOT ENCODING
        # ====================================================

        item_type_columns = [

            "Item_Type_Breads",

            "Item_Type_Breakfast",

            "Item_Type_Canned",

            "Item_Type_Dairy",

            "Item_Type_Frozen Foods",

            "Item_Type_Fruits and Vegetables",

            "Item_Type_Hard Drinks",

            "Item_Type_Health and Hygiene",

            "Item_Type_Household",

            "Item_Type_Meat",

            "Item_Type_Others",

            "Item_Type_Seafood",

            "Item_Type_Snack Foods",

            "Item_Type_Soft Drinks",

            "Item_Type_Starchy Foods"
        ]


        # Create all dummy columns with 0

        for col in item_type_columns:

            input_data[col] = 0


        # Set selected item type to 1

        selected_item_col = (
            "Item_Type_" + item_type
        )


        if selected_item_col in input_data.columns:

            input_data[selected_item_col] = 1


        # ====================================================
        # OUTLET IDENTIFIER ONE-HOT ENCODING
        # ====================================================

        outlet_columns = [

            "Outlet_Identifier_OUT013",

            "Outlet_Identifier_OUT017",

            "Outlet_Identifier_OUT018",

            "Outlet_Identifier_OUT019",

            "Outlet_Identifier_OUT027",

            "Outlet_Identifier_OUT035",

            "Outlet_Identifier_OUT045",

            "Outlet_Identifier_OUT046",

            "Outlet_Identifier_OUT049"
        ]


        # Create all dummy columns with 0

        for col in outlet_columns:

            input_data[col] = 0


        # Set selected outlet to 1

        selected_outlet = (
            "Outlet_Identifier_" +
            outlet_identifier
        )


        if selected_outlet in input_data.columns:

            input_data[selected_outlet] = 1


        # ====================================================
        # OUTLET TYPE ONE-HOT ENCODING
        # ====================================================

        outlet_type_columns = [

            "Outlet_Type_Supermarket Type1",

            "Outlet_Type_Supermarket Type2",

            "Outlet_Type_Supermarket Type3"
        ]


        # Create all dummy columns with 0

        for col in outlet_type_columns:

            input_data[col] = 0


        # Set selected outlet type to 1

        selected_type = (
            "Outlet_Type_" +
            outlet_type
        )


        if selected_type in input_data.columns:

            input_data[selected_type] = 1


        # ====================================================
        # ARRANGE COLUMNS IN TRAINING ORDER
        # ====================================================

        if hasattr(model, "feature_names_in_"):

            training_columns = model.feature_names_in_

            input_data = input_data.reindex(
                columns=training_columns,
                fill_value=0
            )

        else:

            print(
                "Warning: model does not have "
                "feature_names_in_"
            )


        # ====================================================
        # DISPLAY INPUT FOR DEBUGGING
        # ====================================================

        print("\nInput data:")
        print(input_data)

        print(
            "\nNumber of input features:",
            input_data.shape[1]
        )


        # ====================================================
        # MAKE PREDICTION
        # ====================================================

        prediction = model.predict(
            input_data
        )[0]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        return render_template(

            "index.html",

            prediction=round(
                float(prediction),
                2
            ),

            error=None
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("\nPrediction Error:")
        print(e)

        return render_template(

            "index.html",

            prediction=None,

            error=str(e)
        )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )