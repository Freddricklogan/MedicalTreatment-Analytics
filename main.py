"""
Cancer Treatment Efficacy Analysis Tool

This script analyzes cancer treatment efficacy data to evaluate the performance
of different cancer drug regimens. It performs statistical analysis and generates
visualizations to help researchers and medical professionals understand treatment outcomes.

Key capabilities:
- Treatment comparison across multiple drug regimens
- Statistical testing for treatment significance
- Tumor volume analysis across time points
- Correlation analysis between tumor metrics and treatment outcomes
- Visualization of key findings and statistical results

Author: Freddrick Logan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path
import matplotlib.ticker as ticker

# Configure visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.1)

def load_study_data(mouse_metadata_path, study_results_path):
    """
    Load and merge mouse metadata and study results data.
    
    Parameters:
        mouse_metadata_path (str): Path to mouse metadata CSV
        study_results_path (str): Path to study results CSV
        
    Returns:
        pandas.DataFrame: Merged dataset of mouse metadata and study results
    """
    print(f"Loading mouse metadata from: {mouse_metadata_path}")
    print(f"Loading study results from: {study_results_path}")
    
    try:
        # Load the mouse data and study results
        mouse_metadata = pd.read_csv(mouse_metadata_path)
        study_results = pd.read_csv(study_results_path)
        
        # Merge the data frames
        merged_data = pd.merge(study_results, mouse_metadata, on="Mouse ID", how="left")
        
        print(f"Successfully loaded and merged data: {len(merged_data)} records")
        return merged_data
    
    except Exception as e:
        print(f"Error loading study data: {e}")
        return None

def clean_data(merged_data):
    """
    Clean the merged dataset by identifying and handling duplicate data.
    
    Parameters:
        merged_data (pandas.DataFrame): Merged dataset to clean
        
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    if merged_data is None:
        print("No data to clean.")
        return None
    
    print("Cleaning dataset...")
    
    # Check for any mouse ID with duplicate time points
    duplicate_mice = merged_data[merged_data.duplicated(subset=["Mouse ID", "Timepoint"])]["Mouse ID"].unique()
    print(f"Found {len(duplicate_mice)} mice with duplicate timepoints: {', '.join(duplicate_mice)}")
    
    # Get all data for duplicate mice
    duplicate_mice_data = merged_data[merged_data["Mouse ID"].isin(duplicate_mice)]
    
    # Display duplicate data for analysis
    print(f"Duplicate data sample:\n{duplicate_mice_data.head()}")
    
    # Create a clean dataframe by dropping mice with duplicate timepoints
    clean_merged_data = merged_data[~merged_data["Mouse ID"].isin(duplicate_mice)]
    
    # Confirm data was cleaned properly
    clean_duplicate_check = clean_merged_data[clean_merged_data.duplicated(subset=["Mouse ID", "Timepoint"])]
    if len(clean_duplicate_check) > 0:
        print("Warning: Duplicates still exist after cleaning.")
    else:
        print("Data cleaning successful.")
    
    # Summarize cleaned data
    mice_count = len(clean_merged_data["Mouse ID"].unique())
    print(f"Total number of mice in cleaned dataset: {mice_count}")
    
    return clean_merged_data

def generate_summary_statistics(clean_data, output_dir):
    """
    Generate summary statistics for tumor volumes by drug regimen.
    
    Parameters:
        clean_data (pandas.DataFrame): Cleaned dataset
        output_dir (str): Directory to save outputs
        
    Returns:
        pandas.DataFrame: Summary statistics by drug regimen
    """
    if clean_data is None:
        print("No data for summary statistics.")
        return None
    
    print("Generating summary statistics by drug regimen...")
    
    # Create a summary statistics dataframe
    summary_stats = pd.DataFrame()
    
    # Dictionary to store statistical results
    regimen_stats = {}
    
    # Calculate summary statistics for each drug regimen
    for regimen in clean_data["Drug Regimen"].unique():
        # Get tumor volumes for this regimen
        regimen_data = clean_data[clean_data["Drug Regimen"] == regimen]["Tumor Volume (mm3)"]
        
        # Calculate statistics
        regimen_stats[regimen] = {
            "Mean": regimen_data.mean(),
            "Median": regimen_data.median(),
            "Variance": regimen_data.var(),
            "Standard Deviation": regimen_data.std(),
            "SEM": stats.sem(regimen_data),
            "Sample Size (n)": len(regimen_data)
        }
    
    # Convert to DataFrame
    summary_stats = pd.DataFrame(regimen_stats).T.round(2)
    
    # Reset index for better display
    summary_stats = summary_stats.reset_index().rename(columns={"index": "Drug Regimen"})
    
    # Sort by mean tumor volume
    summary_stats = summary_stats.sort_values("Mean")
    
    # Display summary statistics
    print("\nSummary Statistics by Drug Regimen:")
    print(summary_stats)
    
    # Save to CSV
    summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    print(f"Summary statistics saved to {os.path.join(output_dir, 'summary_statistics.csv')}")
    
    return summary_stats

def create_bar_charts(clean_data, output_dir):
    """
    Create bar charts of mouse count by drug regimen.
    
    Parameters:
        clean_data (pandas.DataFrame): Cleaned dataset
        output_dir (str): Directory to save outputs
    """
    if clean_data is None:
        print("No data for creating bar charts.")
        return
    
    print("Creating bar charts of mouse count by drug regimen...")
    
    # Count mice per drug regimen
    mice_per_regimen = clean_data.groupby("Drug Regimen")["Mouse ID"].nunique().sort_values(ascending=False)
    
    # Generate bar chart using Pandas plot
    plt.figure(figsize=(12, 6))
    mice_per_regimen.plot(kind="bar", color="skyblue")
    plt.title("Number of Mice Tested per Drug Regimen", fontsize=14)
    plt.xlabel("Drug Regimen", fontsize=12)
    plt.ylabel("Number of Mice", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mice_per_regimen_pandas.png"))
    plt.close()
    
    # Generate bar chart using pyplot
    plt.figure(figsize=(12, 6))
    plt.bar(mice_per_regimen.index, mice_per_regimen.values, color="lightcoral")
    plt.title("Number of Mice Tested per Drug Regimen", fontsize=14)
    plt.xlabel("Drug Regimen", fontsize=12)
    plt.ylabel("Number of Mice", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mice_per_regimen_pyplot.png"))
    plt.close()
    
    print("Bar charts created successfully.")

def create_pie_charts(clean_data, output_dir):
    """
    Create pie charts of mice distribution by sex.
    
    Parameters:
        clean_data (pandas.DataFrame): Cleaned dataset
        output_dir (str): Directory to save outputs
    """
    if clean_data is None:
        print("No data for creating pie charts.")
        return
    
    print("Creating pie charts of mice distribution by sex...")
    
    # Get unique mice
    unique_mice = clean_data.drop_duplicates(subset=["Mouse ID"])
    
    # Count mice by sex
    mice_by_sex = unique_mice["Sex"].value_counts()
    
    # Create Pandas pie chart
    plt.figure(figsize=(8, 8))
    mice_by_sex.plot(kind="pie", autopct="%1.1f%%", colors=["lightblue", "lightpink"], 
                     shadow=True, startangle=140, explode=(0.05, 0))
    plt.title("Mice Distribution by Sex", fontsize=14)
    plt.ylabel("")  # Hide the ylabel
    plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mice_by_sex_pandas.png"))
    plt.close()
    
    # Create pyplot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(mice_by_sex, labels=mice_by_sex.index, autopct="%1.1f%%", 
            colors=["lightblue", "lightpink"], shadow=True, startangle=140, 
            explode=(0.05, 0))
    plt.title("Mice Distribution by Sex", fontsize=14)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mice_by_sex_pyplot.png"))
    plt.close()
    
    print("Pie charts created successfully.")

def calculate_quartiles_and_outliers(clean_data, output_dir):
    """
    Calculate quartiles, IQR, and identify potential outliers.
    
    Parameters:
        clean_data (pandas.DataFrame): Cleaned dataset
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: A tuple containing outlier results and final timepoint tumor data
    """
    if clean_data is None:
        print("No data for quartile calculations.")
        return None, None
    
    print("Calculating quartiles and identifying outliers...")
    
    # Select the 4 treatment regimens of interest
    treatment_regimens = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]
    
    # Get the maximum timepoint for each mouse
    max_timepoint_per_mouse = clean_data.groupby(["Mouse ID"])["Timepoint"].max().reset_index()
    
    # Merge with the original dataframe to get the final tumor volume
    final_timepoint_data = max_timepoint_per_mouse.merge(
        clean_data, on=["Mouse ID", "Timepoint"], how="left"
    )
    
    # Filter for the treatment regimens of interest
    final_timepoint_data = final_timepoint_data[
        final_timepoint_data["Drug Regimen"].isin(treatment_regimens)
    ]
    
    # Create a summary table for quartile info
    quartile_summary = []
    
    # Loop through the treatment regimens to calculate quartiles
    for regimen in treatment_regimens:
        # Get the tumor volumes for this regimen
        regimen_tumor_volumes = final_timepoint_data[
            final_timepoint_data["Drug Regimen"] == regimen
        ]["Tumor Volume (mm3)"]
        
        # Calculate quartiles and IQR
        q1 = regimen_tumor_volumes.quantile(0.25)
        q3 = regimen_tumor_volumes.quantile(0.75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find potential outliers
        outliers = regimen_tumor_volumes[
            (regimen_tumor_volumes < lower_bound) | (regimen_tumor_volumes > upper_bound)
        ]
        
        # Add to summary
        quartile_summary.append({
            "Regimen": regimen,
            "Q1": q1,
            "Median": regimen_tumor_volumes.median(),
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": len(outliers),
            "Outlier Values": outliers.tolist()
        })
    
    # Convert summary to DataFrame
    quartile_df = pd.DataFrame(quartile_summary)
    
    # Display quartile info
    print("\nQuartile Analysis Summary:")
    print(quartile_df[["Regimen", "Q1", "Median", "Q3", "IQR", "Outlier Count"]])
    
    # Save quartile summary to CSV
    quartile_df.to_csv(os.path.join(output_dir, "quartile_analysis.csv"), index=False)
    print(f"Quartile analysis saved to {os.path.join(output_dir, 'quartile_analysis.csv')}")
    
    return quartile_df, final_timepoint_data

def create_box_plots(quartile_data, final_timepoint_data, output_dir):
    """
    Create box plots of final tumor volume for treatment regimens.
    
    Parameters:
        quartile_data (pandas.DataFrame): DataFrame with quartile information
        final_timepoint_data (pandas.DataFrame): DataFrame with final timepoint data
        output_dir (str): Directory to save outputs
    """
    if quartile_data is None or final_timepoint_data is None:
        print("No data for creating box plots.")
        return
    
    print("Creating box plots of final tumor volume...")
    
    # Set up figure
    plt.figure(figsize=(14, 8))
    
    # Create box plot
    box_plot = sns.boxplot(x="Drug Regimen", y="Tumor Volume (mm3)", 
                          data=final_timepoint_data[final_timepoint_data["Drug Regimen"].isin(quartile_data["Regimen"])],
                          palette="Set3")
    
    # Customize plot
    plt.title("Final Tumor Volume by Drug Regimen", fontsize=16)
    plt.xlabel("Drug Regimen", fontsize=14)
    plt.ylabel("Tumor Volume (mm3)", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    
    # Add custom styling
    for i, box in enumerate(box_plot.artists):
        box.set_edgecolor("black")
        box.set_linewidth(2)
        
        # Change the color of whiskers and caps
        for j in range(6 * i, 6 * (i + 1)):
            if j % 6 < 2:  # Caps
                box_plot.lines[j].set_color("black")
                box_plot.lines[j].set_linewidth(2)
            else:  # Whiskers and median line
                box_plot.lines[j].set_color("black")
                box_plot.lines[j].set_linewidth(2)
    
    # Annotate outliers
    for regimen in quartile_data["Regimen"]:
        regimen_data = quartile_data[quartile_data["Regimen"] == regimen]
        if regimen_data["Outlier Count"].values[0] > 0:
            outlier_values = regimen_data["Outlier Values"].values[0]
            for outlier in outlier_values:
                plt.annotate(f"{outlier:.2f}", 
                           xy=(regimen, outlier),
                           xytext=(10, 0),
                           textcoords="offset points",
                           fontsize=8, color="red",
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_tumor_volume_boxplot.png"))
    plt.close()
    
    print("Box plots created successfully.")

def create_line_and_scatter_plots(clean_data, output_dir):
    """
    Create line and scatter plots for tumor volume vs. time point.
    
    Parameters:
        clean_data (pandas.DataFrame): Cleaned dataset
        output_dir (str): Directory to save outputs
    """
    if clean_data is None:
        print("No data for creating line and scatter plots.")
        return
    
    print("Creating line and scatter plots...")
    
    # Select a mouse treated with Capomulin
    capomulin_mice = clean_data[clean_data["Drug Regimen"] == "Capomulin"]["Mouse ID"].unique()
    selected_mouse = capomulin_mice[0]  # Take the first mouse for demonstration
    
    # Get data for selected mouse
    mouse_data = clean_data[clean_data["Mouse ID"] == selected_mouse]
    
    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(mouse_data["Timepoint"], mouse_data["Tumor Volume (mm3)"], 
             marker="o", linestyle="-", linewidth=2, color="blue")
    plt.title(f"Capomulin Treatment: Mouse {selected_mouse}", fontsize=14)
    plt.xlabel("Timepoint (Days)", fontsize=12)
    plt.ylabel("Tumor Volume (mm3)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(mouse_data["Timepoint"].unique())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"mouse_{selected_mouse}_line_plot.png"))
    plt.close()
    
    # Create a scatter plot of tumor volume vs. mouse weight for Capomulin treatment
    capomulin_data = clean_data[clean_data["Drug Regimen"] == "Capomulin"]
    
    # Get average tumor volume for each mouse
    avg_tumor_vol = capomulin_data.groupby("Mouse ID")["Tumor Volume (mm3)"].mean()
    
    # Get mouse weight (should be the same for all observations of a single mouse)
    mouse_weights = capomulin_data.drop_duplicates("Mouse ID").set_index("Mouse ID")["Weight (g)"]
    
    # Combine into a dataframe
    regression_data = pd.DataFrame({
        "Weight": mouse_weights,
        "Avg Tumor Volume": avg_tumor_vol
    })
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        regression_data["Weight"], regression_data["Avg Tumor Volume"]
    )
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot scatter points
    plt.scatter(regression_data["Weight"], regression_data["Avg Tumor Volume"], 
                color="blue", edgecolor="black", alpha=0.7, s=80)
    
    # Plot regression line
    x_values = np.array([min(regression_data["Weight"]), max(regression_data["Weight"])])
    y_values = intercept + slope * x_values
    plt.plot(x_values, y_values, color="red", linestyle="--", linewidth=2,
             label=f"y = {slope:.2f}x + {intercept:.2f} (r² = {r_value**2:.2f})")
    
    # Add labels and title
    plt.title("Average Tumor Volume vs. Mouse Weight\nCapomulin Treatment", fontsize=14)
    plt.xlabel("Weight (g)", fontsize=12)
    plt.ylabel("Average Tumor Volume (mm3)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_vs_tumor_volume.png"))
    plt.close()
    
    print("Line and scatter plots created successfully.")

def perform_statistical_analysis(clean_data, output_dir):
    """
    Perform statistical analysis on treatment efficacy.
    
    Parameters:
        clean_data (pandas.DataFrame): Cleaned dataset
        output_dir (str): Directory to save outputs
    """
    if clean_data is None:
        print("No data for statistical analysis.")
        return
    
    print("Performing statistical analysis...")
    
    # Treatment regimens to compare
    treatment_regimens = ["Capomulin", "Ramicane", "Infubinol", "Ceftamin"]
    
    # Get the maximum timepoint for each mouse
    max_timepoint_per_mouse = clean_data.groupby(["Mouse ID"])["Timepoint"].max().reset_index()
    
    # Merge with the original dataframe to get the final tumor volume
    final_timepoint_data = max_timepoint_per_mouse.merge(
        clean_data, on=["Mouse ID", "Timepoint"], how="left"
    )
    
    # Filter for the treatment regimens of interest
    final_timepoint_data = final_timepoint_data[
        final_timepoint_data["Drug Regimen"].isin(treatment_regimens)
    ]
    
    # Create a dictionary to store p-values
    p_values = {}
    
    # Use Capomulin as the reference treatment
    reference_treatment = "Capomulin"
    reference_data = final_timepoint_data[
        final_timepoint_data["Drug Regimen"] == reference_treatment
    ]["Tumor Volume (mm3)"]
    
    # Compare reference treatment to each other treatment
    for treatment in treatment_regimens:
        if treatment != reference_treatment:
            # Get treatment data
            treatment_data = final_timepoint_data[
                final_timepoint_data["Drug Regimen"] == treatment
            ]["Tumor Volume (mm3)"]
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(reference_data, treatment_data, equal_var=False)
            
            # Store p-value
            p_values[treatment] = p_val
    
    # Create a dataframe to display results
    p_values_df = pd.DataFrame(list(p_values.items()), columns=["Treatment", "p-value vs. Capomulin"])
    
    # Add significance indicator
    p_values_df["Significant (p < 0.05)"] = p_values_df["p-value vs. Capomulin"] < 0.05
    
    # Sort by p-value
    p_values_df = p_values_df.sort_values("p-value vs. Capomulin")
    
    # Display results
    print("\nStatistical Analysis Results:")
    print(p_values_df)
    
    # Save to CSV
    p_values_df.to_csv(os.path.join(output_dir, "statistical_analysis.csv"), index=False)
    print(f"Statistical analysis saved to {os.path.join(output_dir, 'statistical_analysis.csv')}")

def main():
    """
    Main function to execute the cancer treatment analysis.
    """
    print("Cancer Treatment Efficacy Analysis Starting...")
    
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mouse_data_path = os.path.join(current_dir, 'data', 'Mouse_metadata.csv')
    study_results_path = os.path.join(current_dir, 'data', 'Study_results.csv')
    output_dir = os.path.join(current_dir, 'analysis')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load study data
    merged_data = load_study_data(mouse_data_path, study_results_path)
    
    if merged_data is None:
        print("Failed to load study data. Exiting program.")
        return
    
    # Clean data
    clean_merged_data = clean_data(merged_data)
    
    if clean_merged_data is None:
        print("Failed to clean data. Exiting program.")
        return
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(clean_merged_data, output_dir)
    
    # Create visualizations
    create_bar_charts(clean_merged_data, output_dir)
    create_pie_charts(clean_merged_data, output_dir)
    
    # Calculate quartiles and identify outliers
    quartile_data, final_timepoint_data = calculate_quartiles_and_outliers(clean_merged_data, output_dir)
    
    # Create box plots
    create_box_plots(quartile_data, final_timepoint_data, output_dir)
    
    # Create line and scatter plots
    create_line_and_scatter_plots(clean_merged_data, output_dir)
    
    # Perform statistical analysis
    perform_statistical_analysis(clean_merged_data, output_dir)
    
    print("Cancer treatment analysis completed successfully!")

if __name__ == "__main__":
    main()