from gurobipy import Model, GRB, quicksum
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os
from datetime import datetime



def model_organize_results(var_values, var_df):
    counter = 0
    for v in var_values:
        # if(v.X>0):
        current_var = re.split("\[|,|]", v.varName)[:-1]
        current_var.append(round(v.X, 4))
        var_df.loc[counter] = current_var
        counter = counter + 1
        # with open("./math_model_outputs/" + 'mip-results.txt',
        #           "w") as f:  # a: open for writing, appending to the end of the file if it exists
        #     f.write(','.join(map(str, current_var)) + '\n')
        # print(','.join(map(str,current_var )))
    return var_df

def normalize_turkish(text):
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'i',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    return text.lower()

# ----------------------------
# EXCEL'DEN PARAMETRELERİN OLDUĞU DOSYAYI OKUMA
# ----------------------------
data_path = "inputs/inputs_to_load.xlsx"

xls = pd.ExcelFile(data_path)

f_k = {"R": 300, "M": 200} # R ve M ekiplerinin sabir yer değiştirme maliyeti

# Sets sayfasından kümeleri çek
df_sets = pd.read_excel(xls, sheet_name="Sets").dropna(how='any')
df_sets["Set"] = df_sets["Set"].astype(str).str.strip()
df_sets["Element"] = df_sets["Element"].astype(str).str.strip()
I = df_sets[df_sets["Set"] == "I"]["Element"].tolist()
J = df_sets[df_sets["Set"] == "J"]["Element"].tolist()
D = pd.to_numeric(df_sets[df_sets["Set"] == "D"]["Element"]).tolist()
K = df_sets[df_sets["Set"] == "K"]["Element"].tolist()

# Diğer parametre sayfaları
swi_df = pd.read_excel(xls, sheet_name="SWIjd").dropna(how='all')
swi_df.columns = swi_df.columns.str.strip()

dij_df = pd.read_excel(xls, sheet_name="dij").dropna(how='all')
dij_df.columns = dij_df.columns.str.strip()

cijk_df = pd.read_excel(xls, sheet_name="cijk").dropna(how='all')
cijk_df.columns = cijk_df.columns.str.strip()



# # Create a small instance for validation purposes
# experiment_supply_cities = ['erzurum', 'konya']
# experiment_demand_cities = ['hatay', 'adiyaman']
# # Filter based on 'i' only
# I = [city for city in I if city in experiment_supply_cities]
# J = [city for city in J if city in experiment_demand_cities]
#
# # For dij_df
# dij_df = dij_df[
#     ((dij_df["i"].isin(experiment_supply_cities)) & (dij_df["j"].isin(experiment_demand_cities))) |
#     ((dij_df["i"].isin(experiment_demand_cities)) & (dij_df["j"].isin(experiment_demand_cities)))
# ]
#
# # For cijk_df
# cijk_df = cijk_df[
#     ((cijk_df["i"].isin(experiment_supply_cities)) & (cijk_df["j"].isin(experiment_demand_cities))) |
#     ((cijk_df["i"].isin(experiment_demand_cities)) & (cijk_df["j"].isin(experiment_demand_cities)))
# ]
#

# ----------------------------
# sidk'nin gün bazlı oransal hesaplanması. tedarik illerinde bulunan R M E değerlerini gün bazlı olarak oransal şekilde dağıttık
# ----------------------------
ratios = {
    1: 0.30,
    2: 0.30,
    3: 0.20,
    4: 0.1,
    5: 0.1
}
total_capacity = {"R": 24000, "M": 18000, "E": 15000}
supply_cities = I

records = []
for city in supply_cities:
    for day in ratios:
        row = {"i": city, "d": day}
        for k in total_capacity:
            row[k] = int(total_capacity[k] * ratios[day])
        records.append(row)

sidk_df = pd.DataFrame(records)
sidk = {(row.i, row.d, k): getattr(row, k) for row in sidk_df.itertuples(index=False) for k in ["R", "M", "E"]}




# ----------------------------
# Parametre sözlükleri
# ----------------------------
cijk_df["i"] = cijk_df["i"]
cijk_df["j"] = cijk_df["j"]
cijk_df["k"] = cijk_df["k"]
cijk = {(row.i, row.j, row.k): row.cost for row in cijk_df.itertuples(index=False)}

SWIjd = {(row.j, row.d): row.SWIjd for row in swi_df.itertuples(index=False)}
dij = {(row.i.strip(), row.j.strip()): row.distance for row in dij_df.itertuples(index=False)}

# ----------------------------
# MODEL INIT
# ----------------------------
model = Model("Disaster_Team_Allocation")



# x = model.addVars(
#     ((i, j, d, k) for (i, j, k) in IJ_valid for d in D),
#     vtype=GRB.CONTINUOUS, name="x"
# )

x_ijdk = model.addVars(
    I, J, D, K,
    vtype=GRB.CONTINUOUS, name="x"
)

y_ijdk =  model.addVars(
    I, J, D, K,
    vtype=GRB.CONTINUOUS, name="f"
)
# model.update()
m_ijjdk = model.addVars(I, J, J, D[:-1], ["R", "M"], vtype=GRB.CONTINUOUS, name="m")

lambda_dk = model.addVars(D, ["R", "M"], vtype=GRB.CONTINUOUS, lb=0.0, name="lambda")




# ----------------------------
# OBJECTIVE FUNCTION
# ----------------------------
# model.update()

assignment_cost = quicksum(
    cijk[i, j, k] * dij[i, j] * y_ijdk[i, j, d, k]
    for i in I for j in J for d in D for k in K
)



# assignment_cost = quicksum(
#     cijk[i, j, k] * dij[i, j] * x[i, j, d, k]
#     for (i, j, k) in IJ_valid for d in D
# )

movement_cost = quicksum(
    (dij[j, jp] + (f_k[k] if j != jp else 0)) * m_ijjdk[i, j, jp, d, k]
    for i in I for j in J for jp in J for d in D[:-1] for k in ["R", "M"]
)

# movement_cost = quicksum(
#     (dij[j, jp] + f_k[k]) * m[i, j, jp, d, k]
#     for i in I for j in J for jp in J for d in D[:-1] for k in ["R", "M"]
#     if (j, jp) in dij
# )

model.setObjective(assignment_cost + movement_cost, GRB.MINIMIZE)



# ----------------------------
# CONSTRAINTS
# ----------------------------
for i in I:  # (2)
    for d in D:
        for k in K:
            cumulative_supply = quicksum(sidk[i, dp, k] for dp in D if dp <= d)
            model.addConstr(
                quicksum(x_ijdk[i, j, d, k] for j in J) == cumulative_supply,
                name=f"supply_cumulative_{i}_{d}_{k}"
            )


for i in I:  # (3)
    for d in D:
        for k in K:
            model.addConstr(
                quicksum(y_ijdk[i, j, d, k] for j in J) == sidk[i, d, k],
                name=f"flows_from_source_{i}_{d}_{k}"
            )
#
# for i in I:  # (3)
#     for d in D[:-1]:
#         for k in ["R", "M"]:
#             model.addConstr(
#                 quicksum(x[i, jp, d + 1, k] for jp in J) ==
#                 quicksum(x[i, j, d, k] for j in J),
#                 name=f"continuity_{i}_{d}_{k}"
#             )

for i in I:  # (4)
    for j in J:
        for d in D[:-1]:
            for k in ["R", "M"]:
                model.addConstr(
                    quicksum(m_ijjdk[i, j, jp, d, k] for jp in J) <= x_ijdk[i, j, d, k],
                    name=f"move_from_presence_{i}_{j}_{d}_{k}"
                    )

for i in I:  # (5 and 6)
    for jp in J:
        for d in D[:-1]:
            # (5)
            for k in ["R", "M"]:
                model.addConstr(
                    x_ijdk[i, jp, d + 1, k] == (quicksum(m_ijjdk[i, j, jp, d, k] for j in J) + y_ijdk[i, jp, d + 1, k]),
                    name=f"move_result_assignment_{i}_{jp}_{d}_{k}"
                )
            # (6)
            k ='E'
            model.addConstr(
                x_ijdk[i, jp, d + 1, k] == (x_ijdk[i, jp, d,  k] + y_ijdk[i, jp, d + 1, k]),
                name=f"move_result_assignment_{i}_{jp}_{d}_{k}"
            )

for j in J:  # (7)
    for d in D:
        for k in ["R", "M"]:
            model.addConstr(
                quicksum(x_ijdk[i, j, d, k] for i in I) == lambda_dk[d,k] * SWIjd[j, d],
                name=f"swi_allocation_{j}_{d}_{k}"
            )

for j in J:  # (8)
    for d in D:
        model.addConstr(
            quicksum(x_ijdk[i, j, d, "R"] for i in I) <=
            3 * quicksum(x_ijdk[i, j, d, "M"] for i in I),
            name=f"medical_support_ratio_{j}_{d}"
        )

for j in J:  # (9)
    for d in D:
        model.addConstr(
            quicksum(x_ijdk[i, j, d, "R"] for i in I ) <=
            5 * quicksum(x_ijdk[i, j, d, "E"] for i in I),
            name=f"equipment_support_ratio_{j}_{d}"
        )



# for j in J:  # (10)
#     for d in D:
#         model.addConstr(
#             quicksum(x[i, j, d, "R"] for i in I if (i, j, "R") in IJ_valid)
#             <= quicksum(x[i, j, d2, "E"] for i in I for d2 in D if d2 <= d and (i, j, "E") in IJ_valid),
#             name=f"order_E_before_R_{j}_{d}"
#         )

# for j in J:  # (11)
#     for d in D:
#         model.addConstr(
#             quicksum(x[i, j, d, "M"] for i in I if (i, j, "M") in IJ_valid)
#             <= quicksum(x[i, j, d2, "R"] for i in I for d2 in D if d2 <= d and (i, j, "R") in IJ_valid),
#             name=f"order_R_before_M_{j}_{d}"
#         )

model.update()
# for constr in model.getConstrs():
#     expr = model.getRow(constr)
#     name = constr.ConstrName
#     rhs = constr.RHS
#     terms = []
#     for i in range(expr.size()):
#         coeff = expr.getCoeff(i)
#         var = expr.getVar(i)
#         terms.append(f"{coeff}*{var.VarName}")
#     lhs_str = " + ".join(terms)
#     print(f"{name}: {lhs_str} == {rhs}")

# ----------------------------
# SOLVE
# ----------------------------
start_time = time.time()
model.optimize()
end_time = time.time()
run_time_cpu = round(end_time - start_time, 2)

if model.status == GRB.INFEASIBLE:
    print("Model infeasible. Computing IIS...")
    model.computeIIS()
    model.write("infeasible.ilp")
    print("Go check infeasible_model.ilp file")
else:
    x_ijdk_results_df = pd.DataFrame(columns=['var_name', 'origin_supply_node', 'demand_node', 'day', 'team_type', 'value'])
    x_ijdk_results_df = model_organize_results(x_ijdk.values(), x_ijdk_results_df)
    x_ijdk_results_df['day'] = x_ijdk_results_df['day'].astype(int)



    y_ijdk_results_df = pd.DataFrame(columns=['var_name', 'from_supply_node', 'to_demand_node', 'day', 'team_type', 'value'])
    y_ijdk_results_df = model_organize_results(y_ijdk.values(), y_ijdk_results_df)
    y_ijdk_results_df['day'] = y_ijdk_results_df['day'].astype(int)

    m_ijjdk_results_df = pd.DataFrame(columns=['var_name', 'origin_supply_node', 'from_demand_node', 'to_demand_node', 'day', 'team_type', 'value'])
    m_ijjdk_results_df = model_organize_results(m_ijjdk.values(), m_ijjdk_results_df)
    m_ijjdk_results_df['day'] = m_ijjdk_results_df['day'].astype(int)

    lambda_dk_results_df = pd.DataFrame(columns=['var_name', 'day', 'team_type', 'value'])
    lambda_dk_results_df = model_organize_results(lambda_dk.values(), lambda_dk_results_df)
    lambda_dk_results_df['day'] = lambda_dk_results_df['day'].astype(int)

# Validation Tests
# Test 1: Validations for Constraints 2 and 3
# Compare available teams of an origin at the network at each day with flows from that origin and actual supply of that origin
# STEP 1: Group total x_ijdk (teams present by origin, day, type)
x_grouped = (
    x_ijdk_results_df
    .groupby(['origin_supply_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={'value': 'total_present'})
)

# STEP 2: Group y_ijdk (new deployments by origin, day, type)
y_grouped = (
    y_ijdk_results_df
    .groupby(['from_supply_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={'from_supply_node': 'origin_supply_node'})
)

# STEP 3: Calculate cumulative supply for each (origin, type)
y_grouped = y_grouped.sort_values(by=['origin_supply_node', 'team_type', 'day'])
y_grouped['cumulative_supply'] = (
    y_grouped
    .groupby(['origin_supply_node', 'team_type'])['value']
    .cumsum()
)


# STEP 4: Calculate cumulative actual supply from sidk_long_df
# Merge with actual s_idk values
sidk_long_df = sidk_df.melt(
    id_vars=['i', 'd'],
    value_vars=['R', 'M', 'E'],
    var_name='team_type',
    value_name='actual_supply'
).rename(columns={'i': 'origin_supply_node', 'd': 'day'})

sidk_long_df = sidk_long_df.sort_values(by=['origin_supply_node', 'team_type', 'day'])
sidk_long_df['cumulative_actual_supply'] = (
    sidk_long_df
    .groupby(['origin_supply_node', 'team_type'])['actual_supply']
    .cumsum()
)


# STEP 4: Merge cumulative supply with actual presence
validation_df = pd.merge(
    x_grouped,
    y_grouped[['origin_supply_node', 'day', 'team_type', 'cumulative_supply']],
    on=['origin_supply_node', 'day', 'team_type'],
    how='left'
)

supply_validation_df = pd.merge(
    validation_df,
    sidk_long_df[['origin_supply_node', 'day', 'team_type', 'cumulative_actual_supply']],
    on=['origin_supply_node', 'day', 'team_type'],
    how='left'
)

# Test 2
# Validation for Constraint 4
# The total number of teams of type k moving from region j on day d,  originally from supply city i, must not exceed the number of teams present at j on that day.
# STEP 1: Group x_ijdk by (origin, from_node, day, team_type)
x_subset = x_ijdk_results_df.rename(columns={'demand_node': 'from_demand_node'})
x_grouped = (
    x_subset
    .groupby(['origin_supply_node', 'from_demand_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={'value': 'teams_present'})
)

# STEP 2: Group m_ijjdk by same keys to get total movements OUT of j
m_grouped = (
    m_ijjdk_results_df
    .groupby(['origin_supply_node', 'from_demand_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={'value': 'teams_moved_out'})
)

# STEP 3: Merge and compare
move_validation_df = pd.merge(
    x_grouped,
    m_grouped,
    on=['origin_supply_node', 'from_demand_node', 'day', 'team_type'],
    how='left'
).fillna(0)

# STEP 4: Calculate difference
move_validation_df['difference'] = move_validation_df['teams_present'] - move_validation_df['teams_moved_out']
move_validation_df['violation'] = move_validation_df['difference'] < -0.001  # flag violations



# Test 3
# Validation for Constraint 5
# The number of teams from source i at region j on day d+1 must equal:
#     + The number of newly deployed teams to j' from source i on day d+1
#     + The number of teams moved into j' from other demand regions at the end of day d

# STEP 1: Filter x for d >= 2 (we are validating x_{i,j,d+1})
x_filtered = x_ijdk_results_df[x_ijdk_results_df['day'] > 1].copy()
x_filtered = x_filtered.rename(columns={'value': 'x_value'})

# STEP 2: Group y_{ij'd+1,k} → deployments from supply i to region j' on day d+1
y_grouped = (
    y_ijdk_results_df
    .rename(columns={'to_demand_node': 'demand_node'})
    .groupby(['from_supply_node', 'demand_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={
        'from_supply_node': 'origin_supply_node',
        'value': 'supplied_from_origin'
    })
)

# STEP 3: Group m_{ij,j',d,k} → movement into region j' at end of day d
m_shifted = m_ijjdk_results_df.rename(columns={'to_demand_node': 'demand_node'})
m_shifted['day'] = m_shifted['day'] + 1  # move from day d to day d+1 (arrival)

m_grouped = (
    m_shifted
    .groupby(['origin_supply_node', 'demand_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={'value': 'moved_from_demand_nodes'})
)


# STEP 4: Merge all into one view
comparison_df = pd.merge(
    x_filtered,
    y_grouped,
    on=['origin_supply_node', 'demand_node', 'day', 'team_type'],
    how='left'
).fillna(0)

comparison_df = pd.merge(
    comparison_df,
    m_grouped,
    on=['origin_supply_node', 'demand_node', 'day', 'team_type'],
    how='left'
).fillna(0)

# Optional: reorder columns
comparison_df = comparison_df[[
    'origin_supply_node', 'demand_node', 'day', 'team_type',
    'x_value', 'supplied_from_origin', 'moved_from_demand_nodes'
]]

# Calculate difference and alarm flag
comparison_df['difference'] = comparison_df['x_value'] - (
    comparison_df['supplied_from_origin'] + comparison_df['moved_from_demand_nodes']
)

comparison_df['violation'] = comparison_df['difference'].abs() > 0.001
presence_flow_validation_df = comparison_df.copy()


# Test 4
# Validate distribution based on SWI
# The total number of teams of type k assigned to region j on day d should be proportional to its SWI score, scaled by λ_dk
# Validate Proportional Allocation via Ratios
# STEP 1: Filter to R and M only
# STEP 1: Filter team types (only Rescue and Medical teams)
x_swi_teams_df = x_ijdk_results_df[x_ijdk_results_df['team_type'].isin(['R', 'M'])].copy()
swi_df = swi_df.rename(columns={'j': 'demand_node', 'd': 'day'})  # unify naming

# STEP 2: Total teams assigned per region/day/team
team_assignment_df = (
    x_swi_teams_df
    .groupby(['demand_node', 'day', 'team_type'])['value']
    .sum()
    .reset_index()
    .rename(columns={'value': 'assigned_teams'})
)

# STEP 3: Add corresponding SWI value for each region/day
assignment_with_swi_df = pd.merge(team_assignment_df, swi_df, on=['demand_node', 'day'], how='left')

# STEP 4: Get total assigned teams and total SWI per day/team type
daily_totals_df = (
    assignment_with_swi_df
    .groupby(['day', 'team_type'])[['assigned_teams', 'SWIjd']]
    .sum()
    .rename(columns={'assigned_teams': 'total_assigned_teams', 'SWIjd': 'total_swi'})
    .reset_index()
)

# STEP 5: Merge totals back into main frame
assignment_with_swi_df = pd.merge(
    assignment_with_swi_df,
    daily_totals_df,
    on=['day', 'team_type'],
    how='left'
)

# STEP 6: Calculate share of teams and share of SWI
assignment_with_swi_df['assigned_ratio'] = assignment_with_swi_df['assigned_teams'] / assignment_with_swi_df['total_assigned_teams']
assignment_with_swi_df['swi_ratio'] = assignment_with_swi_df['SWIjd'] / assignment_with_swi_df['total_swi']

# STEP 7: Compare ratios
assignment_with_swi_df['difference'] = assignment_with_swi_df['assigned_ratio'] - assignment_with_swi_df['swi_ratio']
assignment_with_swi_df['violation'] = assignment_with_swi_df['difference'].abs() > 0.01





# print results to output file

optimization_results_df = pd.DataFrame(
    columns=['model_obj_value', 'model_obj_bound', 'gurobi_time', 'python_time'])
optimization_results_df.loc[len(optimization_results_df.index)] = [model.objval, model.objbound, model.runtime, run_time_cpu]

writer_file_name = os.path.join('outputs', "result_of_run_{}.xlsx".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M'))))

writer = pd.ExcelWriter(writer_file_name)
optimization_results_df.to_excel(writer, sheet_name='optimization_results')
x_ijdk_results_df.to_excel(writer, sheet_name='x_ijdk_results')
y_ijdk_results_df.to_excel(writer, sheet_name='y_ijdk_results')
m_ijjdk_results_df.to_excel(writer, sheet_name='m_ijjdk_results')
lambda_dk_results_df.to_excel(writer, sheet_name='lambda_dk_results')
supply_validation_df.to_excel(writer, sheet_name='supply_validation_df')
move_validation_df.to_excel(writer, sheet_name='move_validation_df')
presence_flow_validation_df.to_excel(writer, sheet_name='presence_flow_validation_df')
assignment_with_swi_df.to_excel(writer, sheet_name='assignment_with_swi_df')
writer.close()

print("All results are printed.")



# Plots

# Filter to top demand nodes for clarity
top_nodes = (
    assignment_with_swi_df.groupby('demand_node')['assigned_teams'].sum()
    .nlargest(10).index
)
plot_df = assignment_with_swi_df[assignment_with_swi_df['demand_node'].isin(top_nodes)]

# Map team type codes to readable names
plot_df['team_type'] = plot_df['team_type'].map({'R': 'Rescue', 'M': 'Medical'})

# Unique team types
team_types = ['Rescue', 'Medical']

# Set up 3x2 grid
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharex=True)
fig.subplots_adjust(hspace=0.35, wspace=0.2)

# Define row titles
row_titles = [
    "Assigned Teams Over Time",
    "Assigned Ratio vs SWI Ratio",
    "Raw SWI Values Over Time"
]

# Loop over team types and fill in the 3 rows
for j, team in enumerate(team_types):
    team_df = plot_df[plot_df['team_type'] == team]

    # --- Row 1: Assigned Teams ---
    sns.lineplot(
        data=team_df,
        x='day', y='assigned_teams',
        hue='demand_node', marker='o',
        ax=axes[0, j]
    )
    axes[0, j].set_title(f"{team}")
    axes[0, j].set_ylabel("Assigned Teams")
    axes[0, j].legend(loc='upper right', fontsize=7)

    # --- Row 2: Assigned Ratio vs SWI Ratio ---
    for node in team_df['demand_node'].unique():
        node_data = team_df[team_df['demand_node'] == node]
        axes[1, j].plot(node_data['day'], node_data['assigned_ratio'], label=f"{node} (assigned)", marker='o')
        axes[1, j].plot(node_data['day'], node_data['swi_ratio'], linestyle='--', label=f"{node} (SWI)", marker='x')
    axes[1, j].set_ylabel("Ratio")
    axes[1, j].legend(loc='upper right', fontsize=7)

    # --- Row 3: Raw SWI Values ---
    sns.lineplot(
        data=team_df,
        x='day', y='SWIjd',
        hue='demand_node', marker='o',
        ax=axes[2, j]
    )
    axes[2, j].set_ylabel("SWI Score")
    axes[2, j].set_xlabel("Day")
    axes[2, j].legend(loc='upper right', fontsize=7)

# Add row labels on the leftmost plots
for i in range(3):
    axes[i, 0].set_ylabel(row_titles[i])

# Add global title
fig.suptitle("Team Allocation Dashboard by Metric and Team Type", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/assigned_teams_plot.png", bbox_inches='tight', dpi=300)


import pandas as pd
import json

# Load from local file
with open('inputs/cities.json', 'r', encoding='utf-8') as f:
    city_data = json.load(f)

# Convert to DataFrame
cities_df = pd.DataFrame(city_data)

cities_df = cities_df[['name', 'latitude', 'longitude']]


# Rename columns for clarity
cities_df = cities_df.rename(columns={
    'name': 'city',
    'latitude': 'lat',
    'longitude': 'lon'
})

cities_df['city'] = cities_df['city'].apply(normalize_turkish)

print(cities_df.head())


# Merge origin city coordinates
x_ijdk_results_df = x_ijdk_results_df.merge(
    cities_df.rename(columns={'city': 'origin_supply_node', 'lat': 'origin_lat', 'lon': 'origin_lon'}),
    on='origin_supply_node',
    how='left'
)

# Merge destination city coordinates
x_ijdk_results_df = x_ijdk_results_df.merge(
    cities_df.rename(columns={'city': 'demand_node', 'lat': 'dest_lat', 'lon': 'dest_lon'}),
    on='demand_node',
    how='left'
)

# Optional: check for missing coordinates
missing = x_ijdk_results_df[x_ijdk_results_df[['origin_lat', 'dest_lat']].isna().any(axis=1)]
if not missing.empty:
    print("Warning: Some cities are missing coordinates:")
    print(missing[['origin_supply_node', 'demand_node']].drop_duplicates())







df_plot = pd.DataFrame(x_ijdk_results_df)

# Convert to GeoDataFrame
gdf_points = gpd.GeoDataFrame(
    df_plot,
    geometry=gpd.points_from_xy(df_plot['dest_lon'], df_plot['dest_lat']),
    crs="EPSG:4326"
)

# Load from local file (adjust path as needed)
countries = gpd.read_file("inputs/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# Filter to Turkey and explode multi-part shapes
turkey = countries[countries['ADMIN'] == 'Turkey'].explode(index_parts=False).reset_index(drop=True)
turkey = turkey.explode(index_parts=False).reset_index(drop=True)

# Plot settings
sns.set(style="white")
days = sorted(gdf_points['day'].unique())
nrows = len(days)

fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 6 * nrows))

# Ensure axes is always iterable
if nrows == 1:
    axes = [axes]

# Define team-specific colors
team_colors = {
    'R': 'red',
    'M': 'blue',
    'E': 'green'
}



# Plot each day
for i, day in enumerate(days):
    ax = axes[i]
    turkey.plot(ax=ax, color='lightgrey', edgecolor='black')

    day_data = gdf_points[gdf_points['day'] == day]
    for team in day_data['team_type'].unique():
        team_data = day_data[day_data['team_type'] == team]
        team_data.plot(
            ax=ax,
            markersize=team_data['value'] / 10,  # Scale for visibility
            alpha=0.6,
            label=f"Team {team}",
            color=team_colors.get(team, 'gray')  # fallback to gray if team unknown
        )

    ax.set_title(f"Team Presence on Day {day}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()

plt.tight_layout()
plt.savefig("outputs/presence_on_map.png", bbox_inches='tight', dpi=300)

# it may be great if we can plot flows on the map too

print("All plots are printed.")