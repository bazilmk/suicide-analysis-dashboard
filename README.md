# Suicide Analysis - Interactive Visualization Dashboard

### Research Study - "Suicide: A concern for the future generations"
- Refer to research report document in repository for detailed study.

### Dataset: ‘Suicide Rates Overview 1991 - 2016’
- Consolidated data released by World Bank, World Health Organization, and United Nations.
- Augmented dataset with more features from World Development Indicators repository.
- Investigated whether there are any significant relationships between specific factors contributing suicide.

## Screenshots

### Dashboard
- Created a single-page interactive visualization dashboard using Dash (Plotly) with multi-select features.
- Dashboard consists of an interactive choropleth map, scatter plot, box-and-whisker plot and parallel coordinate plot.
- All plots are updated simulataneously based on click based selection / interaction on any one plot and/or from adjusting the header options.

![Dashboard](images/dashboard.png?raw=true)


### Dashboard Demo

![Dashboard_Demo](images/dashboard_demo.gif?raw=true)

### Components within the dashboard

#### Header
- Dropdowns, multi-selects, checkboxes and yearly timeline.
- For multi-selects, checkboxes and yearly timeline: All options are pre-selected to load the entire data once the dashboard is opened.

![Header](images/header.png?raw=true)

#### World Map
- All countries pre-selected at the start.

![World Map](images/world_map.png?raw=true)

#### Parallel Coordinate Plot
- Parallel coordinate plot interacts with the world map based on the area selected on the world map.

![Parallel_Coordinate_Plot](images/parallel_coordinate_plot.png?raw=true)

### Scatter Plot
![Scatter_Plot](images/scatter_plot.png?raw=true)

#### Box Plot
![Box_Plot](images/box_plot.png?raw=true)


### Interactive updates across visualizations (Example):
- Selecting a few countries using box select

![Selected map](images/selected_world_map.png?raw=true)

- The other 3 visualizations are filtered based on the selected countries:

![Selected coordinate_plot](images/updated_parallel_coordinate_plot.png?raw=true)
![Selected scatter_plot](images/updates_scatter_plot.png?raw=true)
![Selected box_plot](images/updated_box_plot.png?raw=true)
