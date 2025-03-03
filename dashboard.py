import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from millify import millify

# Set up the basic page configuration
st.set_page_config(
    page_title="Final Project Data Dashboard",
    page_icon="📊",
    layout="wide",
)


def reverse_mapping(mapping_dict, list_to_map):
    """
    Create the mapped list along mapping_dict with :
     - mapping_dict { key: item }
     - list_to_map a list of items
    """
    # Create reverse dict {item: key}
    inverse_mapping = {value: key for key, value in mapping_dict.items()}
    # Retourne la liste des clés correspondantes pour chaque valeur présente dans values_list
    return [inverse_mapping[item] for item in list_to_map if item in inverse_mapping]


# Load data
@st.cache_data
def load_data(filepath):
    data = pd.read_excel(filepath)

    # Get data types neat and clean
    experience_levels_order = ["EN", "MI", "SE", "EX"]
    data["experience_level"] = pd.Categorical(
        data["experience_level"], categories=experience_levels_order, ordered=True
    )

    employment_types = ["PT", "FT", "CT", "FL"]
    data["employment_type"] = pd.Categorical(
        data["employment_type"], categories=employment_types
    )

    company_sizes_order = ["S", "M", "L"]
    data["company_size"] = pd.Categorical(
        data["company_size"], categories=company_sizes_order, ordered=True
    )
    return data


data = load_data("./salaries_data.xlsx")

# Set up traduction
_, language_column = st.columns([6, 1])
with language_column:
    selected_language = st.selectbox(
        "Language",
        options=["\U0001f1ec\U0001f1e7", "\U0001f1eb\U0001f1f7"],
        index=0,
        label_visibility="hidden",
    )

    filename = {
        "\U0001f1ec\U0001f1e7": "./en.json",
        "\U0001f1eb\U0001f1f7": "./fr.json",
    }

    with open(filename[selected_language], "r", encoding="utf-8") as f:
        TRAD = json.load(f)

# Set up header
st.header(TRAD["TITLE"])
st.write(TRAD["TITLE_DESCRIPTION"])

## Set up tabs
tabs = st.tabs(
    [
        TRAD["TABS"]["SALARY"],
        TRAD["TABS"]["TRENDING_JOBS"],
        TRAD["TABS"]["BEST_PAID_JOBS"],
        TRAD["TABS"]["MAPS"],
    ]
)

# region Salary tab - Set up
with tabs[0]:
    st.write(TRAD["SALARY"]["DESCRIPTION"])
    salary_filters_column, salary_data_column = st.columns([1, 3], gap="medium")

    # Set up filters
    filtered_data = data[
        [
            "salary_in_usd",
            "company_location",
            "employment_type",
            "company_size",
            "remote_ratio",
            "job_title",
            "experience_level",
        ]
    ]
    with salary_filters_column:
        st.subheader(TRAD["FILTERS"]["TITLE"])

        # Job title
        ordered_job_titles_avail = sorted(filtered_data["job_title"].unique())
        selected_job_title = st.multiselect(
            TRAD["FILTERS"]["JOB_TITLE"],
            ordered_job_titles_avail,
            placeholder=TRAD["FILTERS"]["ALL"],
            key="salary_job_title_filter",
        )
        if len(selected_job_title) > 0:
            filtered_data = filtered_data[
                filtered_data["job_title"].isin(selected_job_title)
            ]

        # Employment type
        order_employment_type_avail = map(
            lambda x: TRAD["EMPLOYMENT_TYPE"][x],
            filtered_data["employment_type"].unique(),
        )
        selected_employment_type = st.pills(
            TRAD["FILTERS"]["EMPLOYMENT_TYPE"],
            order_employment_type_avail,
            selection_mode="multi",
            key="salary_employment_type_filter",
        )
        if len(selected_employment_type) > 0:
            filtered_data = filtered_data[
                filtered_data["employment_type"].isin(
                    reverse_mapping(TRAD["EMPLOYMENT_TYPE"], selected_employment_type)
                )
            ]

        # Experience level
        ordered_experience_level_avail = map(
            lambda x: TRAD["EXPERIENCE_LEVEL"][x],
            filtered_data["experience_level"].sort_values().unique(),
        )
        selected_experience_level = st.pills(
            TRAD["FILTERS"]["EXPERIENCE_LEVEL"],
            ordered_experience_level_avail,
            selection_mode="multi",
            key="salary_experience_level_filter",
        )
        if len(selected_experience_level) > 0:
            filtered_data = filtered_data[
                filtered_data["experience_level"].isin(
                    reverse_mapping(TRAD["EXPERIENCE_LEVEL"], selected_experience_level)
                )
            ]

        # Remote ratio
        ordered_remote_ratio_avail = map(
            lambda x: TRAD["REMOTE_RATIO"][str(x)],
            filtered_data["remote_ratio"].sort_values().unique(),
        )
        selected_remote_ratio = st.pills(
            TRAD["FILTERS"]["REMOTE_RATIO"],
            ordered_remote_ratio_avail,
            selection_mode="multi",
            key="salary_remote_ratio_filter",
        )
        if len(selected_remote_ratio) > 0:
            selected_remote_ratio = [
                int(x)
                for x in reverse_mapping(TRAD["REMOTE_RATIO"], selected_remote_ratio)
            ]
            filtered_data = filtered_data[
                filtered_data["remote_ratio"].isin(selected_remote_ratio)
            ]

        # Company size
        ordered_company_size_avail = (
            filtered_data["company_size"].sort_values().unique()
        )
        selected_company_size = st.pills(
            TRAD["FILTERS"]["COMPANY_SIZE"],
            ordered_company_size_avail,
            selection_mode="multi",
            key="salary_company_size_filter",
        )
        if len(selected_company_size) > 0:
            filtered_data = filtered_data[
                filtered_data["company_size"].isin(selected_company_size)
            ]

        # Company location
        company_location_avail = map(
            lambda x: TRAD["COUNTRY"][x], filtered_data["company_location"].unique()
        )
        selected_company_location = st.multiselect(
            TRAD["FILTERS"]["COMPANY_LOCATION"],
            sorted(company_location_avail),
            placeholder=TRAD["FILTERS"]["ALL"],
            key="salary_company_location_filter",
        )
        if len(selected_company_location) > 0:
            filtered_data = filtered_data[
                filtered_data["company_location"].isin(
                    reverse_mapping(TRAD["COUNTRY"], selected_company_location)
                )
            ]

    with salary_data_column:
        ## Metrics
        (
            number_column,
            mean_salary_column,
            median_salary_column,
            number_jobs_column,
            number_location_column,
        ) = st.columns([1, 1, 1, 1, 1])
        with number_column:
            st.metric(TRAD["METRIC"]["NUMBER_ROWS"], filtered_data.shape[0])
        with mean_salary_column:
            st.metric(
                TRAD["METRIC"]["MEAN_USD_SALARY"],
                "$" + millify(filtered_data["salary_in_usd"].mean()),
            )
        with median_salary_column:
            st.metric(
                TRAD["METRIC"]["MEDIAN_USD_SALARY"],
                "$" + millify(filtered_data["salary_in_usd"].median()),
            )
        with number_jobs_column:
            st.metric(
                TRAD["METRIC"]["NUMBER_JOBS"],
                filtered_data["job_title"].nunique(),
            )
        with number_location_column:
            st.metric(
                TRAD["METRIC"]["NUMBER_LOCATION"],
                filtered_data["company_location"].nunique(),
            )

        ## Boxplot of salary range
        salary_fig = px.box(
            filtered_data,
            x="salary_in_usd",
            title=TRAD["SALARY"]["TITLE"],
            color_discrete_sequence=["#c06666"],
        )
        salary_fig.update_layout(xaxis_title=TRAD["SALARY"]["XLABEL"])
        salary_fig.update_xaxes(
            showline=True, linewidth=1, linecolor="black", mirror=False
        )
        salary_fig.update_yaxes(
            showline=True, linewidth=1, linecolor="black", mirror=False
        )

        st.plotly_chart(salary_fig)

        # Add quick description
        Q1 = filtered_data["salary_in_usd"].quantile(0.25)
        Q3 = filtered_data["salary_in_usd"].quantile(0.75)
        q1_formatted = f"{Q1:,.0f}"
        q3_formatted = f"{Q3:,.0f}"
        st.markdown(TRAD["SALARY"]["FOOTER"].format(q1=q1_formatted, q3=q3_formatted))

# region Trending jobs - Set up
with tabs[1]:
    st.write(TRAD["TRENDING_JOB"]["DESCRIPTION"])
    trending_job_filters_column, trending_job_data_column = st.columns(
        [1, 3], gap="medium"
    )

    # Set up filters
    filtered_data = data[
        [
            "work_year",
            "salary_in_usd",
            "company_location",
            "employment_type",
            "company_size",
            "remote_ratio",
            "job_title",
            "experience_level",
        ]
    ]
    # Keep last year
    last_year = filtered_data["work_year"].max()
    filtered_data = filtered_data[filtered_data["work_year"] == last_year]

    with trending_job_filters_column:
        st.subheader(TRAD["FILTERS"]["TITLE"])

        # Select number of jobs to plot
        top_x = st.slider(
            TRAD["FILTERS"]["NUMBER_SELECT"],
            min_value=1,
            max_value=filtered_data["job_title"].nunique(),
            value=5,
            key="trend_number_slider",
        )

        # Employment type
        order_employment_type_avail = map(
            lambda x: TRAD["EMPLOYMENT_TYPE"][x],
            filtered_data["employment_type"].unique(),
        )
        selected_employment_type = st.pills(
            TRAD["FILTERS"]["EMPLOYMENT_TYPE"],
            order_employment_type_avail,
            selection_mode="multi",
            key="trend_employment_type_filter",
        )
        if len(selected_employment_type) > 0:
            filtered_data = filtered_data[
                filtered_data["employment_type"].isin(
                    reverse_mapping(TRAD["EMPLOYMENT_TYPE"], selected_employment_type)
                )
            ]

        # Experience level
        ordered_experience_level_avail = map(
            lambda x: TRAD["EXPERIENCE_LEVEL"][x],
            filtered_data["experience_level"].sort_values().unique(),
        )
        selected_experience_level = st.pills(
            TRAD["FILTERS"]["EXPERIENCE_LEVEL"],
            ordered_experience_level_avail,
            selection_mode="multi",
            key="trend_experience_level_filter",
        )
        if len(selected_experience_level) > 0:
            filtered_data = filtered_data[
                filtered_data["experience_level"].isin(
                    reverse_mapping(TRAD["EXPERIENCE_LEVEL"], selected_experience_level)
                )
            ]

        # Remote ratio
        ordered_remote_ratio_avail = map(
            lambda x: TRAD["REMOTE_RATIO"][str(x)],
            filtered_data["remote_ratio"].sort_values().unique(),
        )
        selected_remote_ratio = st.pills(
            TRAD["FILTERS"]["REMOTE_RATIO"],
            ordered_remote_ratio_avail,
            selection_mode="multi",
            key="trend_remote_ratio_filter",
        )
        if len(selected_remote_ratio) > 0:
            selected_remote_ratio = [
                int(x)
                for x in reverse_mapping(TRAD["REMOTE_RATIO"], selected_remote_ratio)
            ]
            filtered_data = filtered_data[
                filtered_data["remote_ratio"].isin(selected_remote_ratio)
            ]

        # Company size
        ordered_company_size_avail = (
            filtered_data["company_size"].sort_values().unique()
        )
        selected_company_size = st.pills(
            TRAD["FILTERS"]["COMPANY_SIZE"],
            ordered_company_size_avail,
            selection_mode="multi",
            key="trend_company_size_filter",
        )
        if len(selected_company_size) > 0:
            filtered_data = filtered_data[
                filtered_data["company_size"].isin(selected_company_size)
            ]

        # Company location
        company_location_avail = map(
            lambda x: TRAD["COUNTRY"][x], filtered_data["company_location"].unique()
        )
        selected_company_location = st.multiselect(
            TRAD["FILTERS"]["COMPANY_LOCATION"],
            sorted(company_location_avail),
            placeholder=TRAD["FILTERS"]["ALL"],
            key="trend_company_location_filter",
        )
        if len(selected_company_location) > 0:
            filtered_data = filtered_data[
                filtered_data["company_location"].isin(
                    reverse_mapping(TRAD["COUNTRY"], selected_company_location)
                )
            ]

    with trending_job_data_column:
        ## Metrics
        number_column, _, height_column = st.columns([1, 2, 1])
        with number_column:
            st.metric(TRAD["METRIC"]["NUMBER_ROWS"], filtered_data.shape[0])

        ## Height slider
        with height_column:
            fig_height = st.slider(
                TRAD["HEIGHT_SLIDER"],
                min_value=300,
                max_value=1200,
                value=600,
                step=10,
                key="trend_height_slider",
            )

        ## Count plot
        job_counts = (
            filtered_data["job_title"].value_counts().nlargest(top_x).reset_index()
        )

        trending_job_fig = px.bar(
            job_counts,
            x="count",
            y="job_title",
            orientation="h",
            title=TRAD["TRENDING_JOB"]["TITLE"].format(n_top=top_x, year=last_year),
            color="job_title",
            height=fig_height,
        )
        trending_job_fig.update_layout(
            xaxis_title=TRAD["TRENDING_JOB"]["XLABEL"].format(year=last_year),
            yaxis_title=TRAD["TRENDING_JOB"]["YLABEL"],
            showlegend=False,
        )
        trending_job_fig.update_xaxes(
            showline=True, linewidth=1, linecolor="black", mirror=False
        )
        trending_job_fig.update_yaxes(
            showline=True, linewidth=1, linecolor="black", mirror=False
        )

        st.plotly_chart(trending_job_fig)
# endregion

# region Best paid jobs - Set up
with tabs[2]:
    st.write(TRAD["BEST_PAID_JOB"]["DESCRIPTION"])
    best_paid_job_filters_column, best_paid_job_data_column = st.columns(
        [1, 3], gap="medium"
    )

    # Set up filters
    filtered_data = data[
        [
            "work_year",
            "salary_in_usd",
            "company_location",
            "employment_type",
            "company_size",
            "remote_ratio",
            "job_title",
            "experience_level",
        ]
    ]
    # Keep last year
    last_year = filtered_data["work_year"].max()
    filtered_data = filtered_data[filtered_data["work_year"] == last_year]

    with best_paid_job_filters_column:
        st.subheader(TRAD["FILTERS"]["TITLE"])

        # Select number of jobs to plot
        top_x = st.slider(
            TRAD["FILTERS"]["NUMBER_SELECT"],
            min_value=1,
            max_value=filtered_data["job_title"].nunique(),
            value=5,
            key="best_paid_slider",
        )

        # Employment type
        order_employment_type_avail = map(
            lambda x: TRAD["EMPLOYMENT_TYPE"][x],
            filtered_data["employment_type"].unique(),
        )
        selected_employment_type = st.pills(
            TRAD["FILTERS"]["EMPLOYMENT_TYPE"],
            order_employment_type_avail,
            selection_mode="multi",
            key="best_paid_employment_type_filter",
        )
        if len(selected_employment_type) > 0:
            filtered_data = filtered_data[
                filtered_data["employment_type"].isin(
                    reverse_mapping(TRAD["EMPLOYMENT_TYPE"], selected_employment_type)
                )
            ]

        # Experience level
        ordered_experience_level_avail = map(
            lambda x: TRAD["EXPERIENCE_LEVEL"][x],
            filtered_data["experience_level"].sort_values().unique(),
        )
        selected_experience_level = st.pills(
            TRAD["FILTERS"]["EXPERIENCE_LEVEL"],
            ordered_experience_level_avail,
            selection_mode="multi",
            key="best_paid_experience_level_filter",
        )
        if len(selected_experience_level) > 0:
            filtered_data = filtered_data[
                filtered_data["experience_level"].isin(
                    reverse_mapping(TRAD["EXPERIENCE_LEVEL"], selected_experience_level)
                )
            ]

        # Remote ratio
        ordered_remote_ratio_avail = map(
            lambda x: TRAD["REMOTE_RATIO"][str(x)],
            filtered_data["remote_ratio"].sort_values().unique(),
        )
        selected_remote_ratio = st.pills(
            TRAD["FILTERS"]["REMOTE_RATIO"],
            ordered_remote_ratio_avail,
            selection_mode="multi",
            key="best_paid_remote_ratio_filter",
        )
        if len(selected_remote_ratio) > 0:
            selected_remote_ratio = [
                int(x)
                for x in reverse_mapping(TRAD["REMOTE_RATIO"], selected_remote_ratio)
            ]
            filtered_data = filtered_data[
                filtered_data["remote_ratio"].isin(selected_remote_ratio)
            ]

        # Company size
        ordered_company_size_avail = (
            filtered_data["company_size"].sort_values().unique()
        )
        selected_company_size = st.pills(
            TRAD["FILTERS"]["COMPANY_SIZE"],
            ordered_company_size_avail,
            selection_mode="multi",
            key="best_paid_company_size_filter",
        )
        if len(selected_company_size) > 0:
            filtered_data = filtered_data[
                filtered_data["company_size"].isin(selected_company_size)
            ]

        # Company location
        company_location_avail = map(
            lambda x: TRAD["COUNTRY"][x], filtered_data["company_location"].unique()
        )
        selected_company_location = st.multiselect(
            TRAD["FILTERS"]["COMPANY_LOCATION"],
            sorted(company_location_avail),
            placeholder=TRAD["FILTERS"]["ALL"],
            key="best_paid_company_location_filter",
        )
        if len(selected_company_location) > 0:
            filtered_data = filtered_data[
                filtered_data["company_location"].isin(
                    reverse_mapping(TRAD["COUNTRY"], selected_company_location)
                )
            ]

    with best_paid_job_data_column:
        ## Metrics
        number_column, _, height_column = st.columns([1, 2, 1])
        with number_column:
            st.metric(TRAD["METRIC"]["NUMBER_ROWS"], filtered_data.shape[0])

        ## Height slider
        with height_column:
            fig_height = st.slider(
                TRAD["HEIGHT_SLIDER"],
                min_value=300,
                max_value=1200,
                value=600,
                step=10,
                key="best_paid_height_slider",
            )

        ## Boxplot
        ordered_jobs = (
            filtered_data.groupby("job_title")["salary_in_usd"]
            .median()
            .sort_values(ascending=False)
            .nlargest(top_x)
            .index.tolist()
        )
        filtered_data = filtered_data[filtered_data["job_title"].isin(ordered_jobs)]

        best_paid_job_fig = px.box(
            filtered_data,
            x="salary_in_usd",
            y="job_title",
            orientation="h",
            title=TRAD["BEST_PAID_JOB"]["TITLE"].format(n_top=top_x, year=last_year),
            color="job_title",
            height=fig_height,
            category_orders={"job_title": ordered_jobs},
        )
        best_paid_job_fig.update_layout(
            xaxis_title=TRAD["BEST_PAID_JOB"]["XLABEL"],
            yaxis_title=TRAD["BEST_PAID_JOB"]["YLABEL"],
            showlegend=False,
        )
        best_paid_job_fig.update_xaxes(
            showline=True, linewidth=1, linecolor="black", mirror=False
        )
        best_paid_job_fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=False,
        )

        st.plotly_chart(best_paid_job_fig)
# endregion

# region Maps - Set up
with tabs[3]:
    st.write(TRAD["MAPS"]["DESCRIPTION"])
    maps_filters_column, maps_data_column = st.columns([1, 3], gap="medium")

    # Set up filters
    filtered_data = data[
        [
            "work_year",
            "salary_in_usd",
            "company_location",
            "employment_type",
            "company_size",
            "remote_ratio",
            "job_title",
            "experience_level",
        ]
    ]

    with maps_filters_column:
        st.subheader(TRAD["MAPS"]["PLOT_CHOICE"])

        ## Select data to plot
        selected_plot = st.radio(
            label=TRAD["FILTERS"]["PLOT_SELECTOR"],
            options=TRAD["MAPS"]["PLOTS"].values(),
        )
        reversed_dict = {v: k for k, v in TRAD["MAPS"]["PLOTS"].items()}
        selected_plot = reversed_dict[selected_plot]

        ## Show available filters
        st.subheader(TRAD["FILTERS"]["TITLE"])

        # Job title - always available
        ordered_job_titles_avail = sorted(filtered_data["job_title"].unique())
        selected_job_title = st.multiselect(
            TRAD["FILTERS"]["JOB_TITLE"],
            ordered_job_titles_avail,
            placeholder=TRAD["FILTERS"]["ALL"],
            key="maps_job_title_filter",
        )
        if len(selected_job_title) > 0:
            filtered_data = filtered_data[
                filtered_data["job_title"].isin(selected_job_title)
            ]

        # Employment type - except when CONTRACT_? is selected
        if selected_plot[:9] != "CONTRACT_":
            order_employment_type_avail = map(
                lambda x: TRAD["EMPLOYMENT_TYPE"][x],
                filtered_data["employment_type"].unique(),
            )
            selected_employment_type = st.pills(
                TRAD["FILTERS"]["EMPLOYMENT_TYPE"],
                order_employment_type_avail,
                selection_mode="multi",
                key="maps_employment_type_filter",
            )
            if len(selected_employment_type) > 0:
                filtered_data = filtered_data[
                    filtered_data["employment_type"].isin(
                        reverse_mapping(
                            TRAD["EMPLOYMENT_TYPE"], selected_employment_type
                        )
                    )
                ]

        # Experience level - always available
        ordered_experience_level_avail = map(
            lambda x: TRAD["EXPERIENCE_LEVEL"][x],
            filtered_data["experience_level"].sort_values().unique(),
        )
        selected_experience_level = st.pills(
            TRAD["FILTERS"]["EXPERIENCE_LEVEL"],
            ordered_experience_level_avail,
            selection_mode="multi",
            key="maps_experience_level_filter",
        )
        if len(selected_experience_level) > 0:
            filtered_data = filtered_data[
                filtered_data["experience_level"].isin(
                    reverse_mapping(TRAD["EXPERIENCE_LEVEL"], selected_experience_level)
                )
            ]

        # Remote ratio - except if REMOTE_? selected
        if selected_plot[:7] != "REMOTE_":
            ordered_remote_ratio_avail = map(
                lambda x: TRAD["REMOTE_RATIO"][str(x)],
                filtered_data["remote_ratio"].sort_values().unique(),
            )
            selected_remote_ratio = st.pills(
                TRAD["FILTERS"]["REMOTE_RATIO"],
                ordered_remote_ratio_avail,
                selection_mode="multi",
                key="maps_remote_ratio_filter",
            )
            if len(selected_remote_ratio) > 0:
                selected_remote_ratio = [
                    int(x)
                    for x in reverse_mapping(
                        TRAD["REMOTE_RATIO"], selected_remote_ratio
                    )
                ]
                filtered_data = filtered_data[
                    filtered_data["remote_ratio"].isin(selected_remote_ratio)
                ]

        # Company size - always available
        ordered_company_size_avail = (
            filtered_data["company_size"].sort_values().unique()
        )
        selected_company_size = st.pills(
            TRAD["FILTERS"]["COMPANY_SIZE"],
            ordered_company_size_avail,
            selection_mode="multi",
            key="maps_company_size_filter",
        )
        if len(selected_company_size) > 0:
            filtered_data = filtered_data[
                filtered_data["company_size"].isin(selected_company_size)
            ]

    with maps_data_column:
        ## Metrics
        number_column, _, height_column = st.columns([1, 2, 1])
        with number_column:
            st.metric(TRAD["METRIC"]["NUMBER_ROWS"], filtered_data.shape[0])

        ## Height slider
        with height_column:
            fig_height = st.slider(
                TRAD["HEIGHT_SLIDER"],
                min_value=300,
                max_value=1200,
                value=600,
                step=10,
                key="maps_height_slider",
            )

        ## Map
        range_value = []
        # Create dataset based on plot
        if selected_plot == "NUMBER_DATA":
            st.warning(TRAD["MAPS"]["DATA_COV_WARNING"])
            map_data = filtered_data["company_location"].value_counts().reset_index()
            map_data_value_column = "count"
            range_value = [
                np.percentile(map_data[map_data_value_column], 5),
                np.percentile(map_data[map_data_value_column], 95),
            ]

        elif selected_plot == "REMOTE_0":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many full office workers in the dataset
            map_data["remote_0"] = filtered_data["remote_ratio"] == 0
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["remote_0"].mean().reset_index()
            )
            map_data["remote_0"] = map_data["remote_0"] * 100
            map_data_value_column = "remote_0"

        elif selected_plot == "REMOTE_50":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many hybrid workers in the dataset
            map_data["remote_50"] = filtered_data["remote_ratio"] == 50
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["remote_50"].mean().reset_index()
            )
            map_data["remote_50"] = map_data["remote_50"] * 100
            map_data_value_column = "remote_50"

        elif selected_plot == "REMOTE_100":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many full remote workers in the dataset
            map_data["remote_100"] = filtered_data["remote_ratio"] == 100
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["remote_100"].mean().reset_index()
            )
            map_data["remote_100"] = map_data["remote_100"] * 100
            map_data_value_column = "remote_100"

        elif selected_plot == "SALARY_MEDIAN":
            map_data = (
                filtered_data.groupby("company_location")["salary_in_usd"]
                .median()
                .reset_index()
            )
            map_data_value_column = "salary_in_usd"

        elif selected_plot == "SALARY_MEAN":
            map_data = (
                filtered_data.groupby("company_location")["salary_in_usd"]
                .mean()
                .reset_index()
            )
            map_data_value_column = "salary_in_usd"

        elif selected_plot == "CONTRACT_FT":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many full remote workers in the dataset
            map_data["ft_contract"] = filtered_data["employment_type"] == "FT"
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["ft_contract"].mean().reset_index()
            )
            map_data["ft_contract"] = map_data["ft_contract"] * 100
            map_data_value_column = "ft_contract"

        elif selected_plot == "CONTRACT_PT":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many full remote workers in the dataset
            map_data["pt_contract"] = filtered_data["employment_type"] == "PT"
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["pt_contract"].mean().reset_index()
            )
            map_data["pt_contract"] = map_data["pt_contract"] * 100
            map_data_value_column = "pt_contract"

        elif selected_plot == "CONTRACT_CT":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many full remote workers in the dataset
            map_data["ct_contract"] = filtered_data["employment_type"] == "CT"
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["ct_contract"].mean().reset_index()
            )
            map_data["ct_contract"] = map_data["ct_contract"] * 100
            map_data_value_column = "ct_contract"

        elif selected_plot == "CONTRACT_FL":
            map_data = pd.DataFrame()
            map_data["company_location"] = filtered_data["company_location"]
            # How many full remote workers in the dataset
            map_data["fl_contract"] = filtered_data["employment_type"] == "FL"
            # Ratio by country
            map_data = (
                map_data.groupby("company_location")["fl_contract"].mean().reset_index()
            )
            map_data["fl_contract"] = map_data["fl_contract"] * 100
            map_data_value_column = "fl_contract"

        # Convert ISO2 into trad country name
        map_data["country_name"] = map_data["company_location"].map(TRAD["COUNTRY"])
        # Convert ISO2 to ISO3
        with open(
            "./country_iso.json",
            "r",
            encoding="utf-8",
        ) as f:
            iso2_to_iso3 = json.load(f)
        map_data["country_iso3"] = map_data["company_location"].map(iso2_to_iso3)

        # Plot it
        map_fig = px.choropleth(
            data_frame=map_data,
            locations="country_iso3",
            locationmode="ISO-3",
            color=map_data_value_column,
            hover_name=map_data["country_name"],
            color_continuous_scale=px.colors.sequential.Plasma,
            projection="natural earth",
            title=TRAD["MAPS"]["PLOTS"][selected_plot] + TRAD["MAPS"]["EACH_COUNTRY"],
            height=fig_height,
            range_color=range_value,
        )

        # Fix legend title
        map_fig.update_layout(
            coloraxis_colorbar=dict(title=TRAD["MAPS"]["LEGEND"][selected_plot])
        )

        st.plotly_chart(map_fig)

        # Add warning
        st.markdown(TRAD["MAPS"]["NOTE"])
