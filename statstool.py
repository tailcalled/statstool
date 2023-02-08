import traceback
import os
import yaml
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import re
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def perform_list_query(query, list):
    mapping = {}
    for hotkey, option, value in list:
        print(f"  {hotkey}) {option}")
        mapping[hotkey] = value
    print()
    print(query)
    choice = input("> ")
    return mapping.get(choice)

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

class State:
    def on_help(self):
        pass
    def on_query(self):
        pass
    def on_after_query(self):
        pass
    def clear_options(self):
        self.options = []
    def open(self):
        return self
    def query(self):
        self.clear_options()
        self.on_help()
        self.on_query()
        print()
        result = perform_list_query("Enter the letter corresponding to your desired command:", self.options)
        print()
        return result
    def add_option(self, hotkey, name, func):
        self.options.append((hotkey, name, func))

class UnloadedState(State):
    def on_query(self):
        self.add_option("l", "Load dataset", self.on_load)
    def on_help(self):
        print("You have not currently loaded any dataset.")
        print("In order to automatically load a dataset, run this program with the following command:")
        print()
        print("python statstool.py <dataset path>")
        print()
        print("Alternative you can use a command among the options below:")
    def on_load(self):
        print("Please enter the file to load:")
        path = input("> ")
        print()
        try:
            path = os.path.abspath(path)
            with open(path, "r") as f:
                print("The file you selected contains:")
                print()
                for line, _ in zip(f, range(5)):
                    print(f"   {line.rstrip()[:80]}...")
                print("    ...")
            print()
            if yn_confirm("Please confirm that this is the intended file."):
                return load_dataset(path)
        except FileNotFoundError:
            print(f"File \"{path}\" not found.")
        except:
            print("Error.")
            traceback.print_exc()

def yn_confirm(query):
    while True:
        print(f"{query} (Y/n)")
        confirmation = input("> ").upper()
        if confirmation == "Y" or confirmation == "":
            return True
        elif confirmation == "N":
            return False
        print("Invalid answer.")

def softmake(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def load_dataset(abspath):
    data = pd.read_csv(abspath)
    config_path = f"{abspath}.config.yaml"
    try:
        return load_project(abspath)
    except FileNotFoundError:
        pass
    print("Project configuration for dataset not found.")
    if yn_confirm("Do you want to create a new project?"):
        output_path = f"{abspath}.output"
        os.makedirs(output_path)
        default_project = {
            "output": os.path.basename(output_path),
            "coding": [],
            "analyses": [],
            "scales": [],
            "last_save": None,
            "autosave": True
        }
        project = LoadedState(abspath, data, config_path, default_project, output_path)
        project.on_save()
        return load_project(abspath)

def load_project(abspath):
    data = pd.read_csv(abspath)
    config_path = f"{abspath}.config.yaml"
    with open(config_path, "r") as f:
        configuration = yaml.load(f.read(), Loader=yaml.SafeLoader)
        output_path = os.path.join(os.path.dirname(abspath), configuration["output"])
    return LoadedState(abspath, data, config_path, configuration, output_path)

poor_chars = re.compile("[^a-zA-Z0-9 ]")
def nice_name(name):
    return re.sub(poor_chars, "", name)
class CancelRun(Exception):
    pass

class LoadedState(State):
    def __init__(self, data_path, data, config_path, configuration, output_path):
        self.base_path = os.path.dirname(data_path)
        self.data_path = data_path
        self.raw_data = data
        print(data)
        self.config_path = config_path
        self.configuration = configuration
        self.output_path = output_path
        softmake(os.path.join(output_path, "backups"))
        softmake(os.path.join(output_path, "analysis"))
        softmake(os.path.join(output_path, "descriptives"))
    def on_help(self):
        print()
        print()
        print(f"Data analysis project in folder: {self.base_path}")
        print(f"Dataset: {os.path.basename(self.data_path)}")
        print(f"Data rows: {len(self.raw_data.index)}")
        print(f"Autosave: {['off', 'on'][self.configuration['autosave']]}")
        if self.configuration["autosave"]:
            print("Note: if you have autosave enabled, you might want to consider going to the backup folder and regularly deleting some of the older backups, as autosave generates a lot of backups.")
        else:
            print(f"Last saved: {self.configuration['last_save']}")
    def on_query(self):
        if not self.configuration["autosave"]:
            self.add_option("s", "Save project", self.on_save)
            self.add_option("a", "Turn on autosave", self.update_config_func("autosave", True))
        else:
            self.add_option("a", "Turn off autosave", self.update_config_func("autosave", False))

        code_menu = MenuState(self)
        self.add_option("c", "Code items", code_menu.open)
        code_menu.add_option("1", "Automatically code items", self.on_autocode)
        if len(self.configuration["coding"]) > 0:
            code_menu.add_option("2", "Update codebooks with missing codes", self.on_update_codebooks)

        analysis_menu = MenuState(self)
        self.add_option("e", "Add analysis", analysis_menu.open)
        factor_analysis_form = FormState(self, self.on_factor_analysis)
        factor_analysis_form.add_property("name", "free", "Name:")
        factor_analysis_form.add_property("items", "items", "Items:")
        factor_analysis_form.add_property("factors", "numbers", "Number of factors:")
        analysis_menu.add_option("f", "Factor analysis", factor_analysis_form.open)
        correlation_matrix_form = FormState(self, self.on_correlation_matrix)
        correlation_matrix_form.add_property("name", "free", "Name:")
        correlation_matrix_form.add_property("variables", "items", "Variables:")
        analysis_menu.add_option("c", "Correlation matrix", correlation_matrix_form.open)

        variable_menu = MenuState(self)
        self.add_option("v", "Add variable", variable_menu.open)
        scale_form = FormState(self, self.on_scale)
        scale_form.add_property("name", "free", "Name:")
        scale_form.add_property("id", "id", "Id:")
        scale_form.add_property("items", "items", "Items:")
        scale_form.add_property("scoring", [("s", "Sum score (pick this if in doubt)", "sum"), ("f", "Factor score (not yet implemented)", "factor")], "Scoring type:")
        variable_menu.add_option("i", "Multi-item scale", scale_form.open)

        self.add_option("r", "Run analyses", self.on_run)
    def on_after_query(self):
        self.autosave()
    def autosave(self):
        if self.configuration["autosave"]:
            self.on_save()
    def update_config_func(self, config, value):
        def on_call():
            self.configuration[config] = value
        return on_call
    def variable_ids(self):
        return set(self.variables_by_id().keys())
    def all_ids(self):
        return self.variable_ids().union(analysis["id"] for analysis in self.configuration["analyses"])
    def allocate_id(self, series):
        ids = self.all_ids()
        for num in range(1000):
            candidate_id = series + str(num).zfill(3)
            if not candidate_id in ids:
                return candidate_id
        raise Exception(f"Out of IDs in series {series}")
    def on_autocode(self):
        print("Automatically coding...")
        precoded = re.compile("-?[0-9]+:.*")
        for column in self.raw_data.columns:
            if not any(coding["raw_name"] == column for coding in self.configuration["coding"]):
                print(f"Coding {column}")
                qid = self.allocate_id("q")
                coding = {
                    "raw_name": column,
                    "coded_ids": [qid]
                }
                if all(precoded.match(value) for value in self.raw_data[column].astype(str)):
                    coding["key"] = {
                        resp: int(resp[:resp.index(":")])
                        for resp in self.raw_data[column].unique()
                    }
                elif str(self.raw_data.dtypes[column]) in ["int", "float", "int64", "float64"]:
                    coding["key"] = "raw"
                else:
                    print(f"Cannot code column {column} of type {self.raw_data.dtypes[column]}")
                    print("Reserving question ID for future manual coding")
                    coding["key"] = "reserved"
                self.configuration["coding"].append(coding)
    def on_update_codebooks(self):
        for coding in self.configuration["coding"]:
            if type(coding["key"]) is dict:
                for resp in self.raw_data[coding["raw_name"]].unique():
                    if resp not in coding["key"]:
                        print(f"Column to code for: {coding['raw_name']}")
                        print(f"Response value to code: {resp}")
                        print(f"Current codebook: {coding['key']}")
                        print("What should the value be coded as?")
                        code = int(input("> ").strip())
                        coding["key"][resp] = code
    def on_factor_analysis(self, settings):
        fid = self.allocate_id("FA")
        self.configuration["analyses"].append({
            "id": fid,
            "name": settings["name"],
            "type": "factor_analysis",
            "items": settings["items"],
            "factors": settings["factors"]
        })
        return self
    def on_correlation_matrix(self, settings):
        cid = self.allocate_id("CM")
        self.configuration["analyses"].append({
            "id": cid,
            "name": settings["name"],
            "type": "correlation_matrix",
            "variables": settings["variables"]
        })
        return self
    def on_scale(self, settings):
        variables_by_id = self.variables_by_id()
        tested_id = self.allocate_id(settings["id"])
        if tested_id != settings["id"] + "000" or tested_id in self.all_ids():
            print("Scale ID is reserved, in use, or otherwise unavailable. Please pick another ID.")
            return
        datasheet, names = self.generate_datasheet(False)
        datasheet, _ = self.score_scales(datasheet, False, names)
        data = datasheet[settings["items"]]
        pca = PCA(1)
        pca.fit(data)
        weights = np.sign(pca.components_[0])
        if settings["scoring"] == "sum":
            print("The items will be scored as follows (in terms of +/-):")
            for i, item in enumerate(settings["items"]):
                if weights[i] == 1:
                    print(f"+ {item} - {variables_by_id[item]}")
            for i, item in enumerate(settings["items"]):
                if weights[i] == -1:
                    print(f"- {item} - {variables_by_id[item]}")
            print()
            answer = perform_list_query("Is this acceptable? (The item scoring can be changed by editing the yaml file, for finer control.)",
                [ ("y", "Yes, create scoring based on this", 1)
                , ("f", "Flip the scoring to be the opposite way, and then it looks good", -1)
                , ("x", "Cancel", None) ]
            )
            if answer is None:
                return self
            scoring = {"weights": [float(x) for x in answer * weights]}
        elif settings["scoring"] == "factor":
            print("Factor scoring is currently not yet implemented.")
            return self.parent
            if not (np.all(weights == 1) or np.all(weights == -1)):
                print("As of now, the factor analysis places the items as opposing ends of this scale:")
                print("End A:")
                for i, item in enumerate(settings["items"]):
                    if weights[i] == 1:
                        print(f"  a) {item} - {variables_by_id[item]}")
                print("End B:")
                for i, item in enumerate(settings["items"]):
                    if weights[i] == -1:
                        print(f"  b) {item} - {variables_by_id[item]}")
                print("The item {item}")
                print("To decide what end of the scale should be +, and what end should be -, we should pick some item to \"define\" the + direction.")
                perform_list_query("What item should define the ")
            scoring = "factor"
        for i in range(len(settings["items"])):
            new_id = self.allocate_id(settings["id"])
            coding = next((coding for coding in self.configuration["coding"] if settings["items"][i] in coding["coded_ids"]), None)
            if coding != None:
                coding["coded_ids"].append(new_id)
            settings["items"][i] = new_id
        self.configuration["scales"].append({
            "name": settings["name"],
            "main_id": settings["id"],
            "all_ids": [settings["id"]],
            "items": settings["items"],
            "scoring": scoring
        })
        return self
    def generate_datasheet(self, expanded):
        data_map = {}
        full_names = {}
        for coding in self.configuration["coding"]:
            if type(coding["key"]) is dict:
                coded = []
                for row in self.raw_data.index:
                    value = self.raw_data[coding["raw_name"]][row]
                    if value not in coding["key"]:
                        print("Response value is missing from codebook. The codebook may be out of date.")
                        print(f"Response value: {value}")
                        print(f"Codebook: {coding['key']}")
                        print("Cancelling run")
                        raise CancelRun("codebook out of date")
                    coded.append(coding["key"][value])
            elif coding["key"] == "raw":
                coded = self.raw_data[coding["raw_name"]].astype(float)
            elif coding["key"] == "reserved":
                continue
            else:
                raise Exception(str(coding))
            for identifier in coding["coded_ids"]:
                full_names[identifier] = f"{identifier} - {coding['raw_name']}" if expanded else identifier
                data_map[full_names[identifier]] = coded
        columns = []
        table = []
        for key in sorted(data_map.keys()):
            table.append(data_map[key])
            columns.append(key)
        datasheet = pd.DataFrame(np.array(table).T, columns=columns)
        return datasheet, full_names
    def score_scales(self, datasheet, expanded, full_names):
        data_map = {}
        reliabilities = {}
        for scale in self.configuration["scales"]:
            if "weights" in scale["scoring"]:
                expected_variance = 0
                value = pd.Series(np.zeros(datasheet.shape[0]), datasheet.index)
                for identifier, weight in zip(scale["items"], scale["scoring"]["weights"]):
                    value += weight * datasheet[full_names[identifier]]
                    expected_variance += (weight * datasheet[full_names[identifier]]).var()
                reliability = (len(scale["items"]) / (len(scale["items"]) - 1)) * (1 - expected_variance / value.var())
            else:
                raise Exception(scale)
            for identifier in scale["all_ids"]:
                full_names[identifier] = f"{identifier} - {scale['name']}" if expanded else identifier
                reliabilities[identifier] = reliability
                data_map[full_names[identifier]] = value
        columns = []
        table = []
        for key in sorted(data_map.keys()):
            table.append(data_map[key])
            columns.append(key)
        scales = pd.DataFrame(np.array(table).T.reshape((len(datasheet.index), len(columns))), columns=columns, index=datasheet.index)
        return pd.concat([scales, datasheet], axis=1), reliabilities
    def on_run(self):
        try:
            print("Coding datasets")
            expanded_datasheet, expanded_names = self.generate_datasheet(True)
            datasheet, short_names = self.generate_datasheet(False)
            print("Scoring scales")
            expanded_datasheet, _ = self.score_scales(expanded_datasheet, True, expanded_names)
            datasheet, reliabilities = self.score_scales(datasheet, False, short_names)
            with open(os.path.join(self.output_path, "reliabilities.txt"), "w") as f:
                f.write("Estimated Cronbach's alpha:\n")
                for scale in reliabilities:
                    f.write(f"{str(np.round(reliabilities[scale], 2)).rjust(6)} {expanded_names[scale]}\n")
            print("Saving datasheets")
            expanded_datasheet.to_csv(os.path.join(self.output_path, "expanded_datasheet.csv"), index=False)
            datasheet.to_csv(os.path.join(self.output_path, "abbreviated_datasheet.csv"), index=False)
            print("Computing descriptives")
            descriptives_path = os.path.join(self.output_path, "descriptives")
            for old_file in os.listdir(descriptives_path):
                os.remove(os.path.join(descriptives_path, old_file))
            for identifier, column in zip(datasheet.columns, expanded_datasheet.columns):
                descriptives = self.compute_descriptives(identifier, expanded_datasheet[column], reliabilities)
                with open(os.path.join(descriptives_path, nice_name(column) + ".txt"), "w") as f:
                    f.write(descriptives)
            print("Performing analyses")
            for analysis in self.configuration["analyses"]:
                print(f"Performing analysis {analysis['name']}")
                path = os.path.join(self.output_path, "analysis", analysis["id"] + " - " + nice_name(analysis["name"]))
                softmake(path)
                for old_file in os.listdir(path):
                    os.remove(os.path.join(path, old_file))
                if analysis["type"] == "factor_analysis":
                    self.perform_factor_analysis(analysis, path, datasheet)
                elif analysis["type"] == "correlation_matrix":
                    self.perform_correlation_matrix(analysis, path, expanded_datasheet, expanded_names)
                else:
                    raise Exception(f"Unrecognized analysis type: {analysis}")
            print(f"Analysis completed - open {self.output_path} to see results")
        except CancelRun as e:
            if str(e) == "codebook out of date":
                if yn_confirm("Do you want to update the codebooks?"):
                    self.on_update_codebooks()
    def perform_factor_analysis(self, analysis, path, datasheet):
        variables_by_id = self.variables_by_id()
        data = datasheet[analysis["items"]]
        fa = FactorAnalyzer(min(20, len(data.columns)-1, len(data.index)-1), rotation=None)
        fa.fit(data)
        variances, _, _ = fa.get_factor_variance()
        plt.plot(np.arange(len(variances))+1, variances, "ro-")
        plt.xlabel("Factor #")
        plt.ylabel("Variance")
        plt.title("Scree plot")
        plt.savefig(os.path.join(path, "scree_plot.png"))
        plt.close()
        lines = []
        lines.append(f"Factor analysis: {analysis['name']}")
        lines.append("")
        lines.append("Scree analysis:")
        for i in range(len(variances)):
            lines.append(f"{str(i+1).rjust(2)}: {np.round(variances[i], 2)}")
        lines.append("")
        lines.append("")
        for n_factors in analysis["factors"]:
            lines.append(f"{n_factors} FACTORS")
            lines.append("===")
            fa = FactorAnalyzer(n_factors, rotation="promax" if n_factors > 1 else None)
            fa.fit(data)
            loadings = fa.loadings_
            for factor in range(n_factors):
                if loadings[np.argmax(np.abs(loadings[:, factor])), factor] < 0:
                    loadings[:, factor] = -loadings[:, factor]
            for factor in range(n_factors):
                lines.append(f"Factor {factor}:")
                for item in np.flip(np.argsort(np.abs(loadings[:, factor]))):
                    if np.all(np.abs(loadings[item, factor]) >= np.abs(loadings[item])):
                        line = ""
                        for loading in loadings[item]:
                            line += str(np.round(loading, 2)).rjust(6) + " "
                        line += variables_by_id[data.columns[item]]
                        lines.append(line)
                lines.append("")
            lines.append("")
        with open(os.path.join(path, "analysis.txt"), "w") as f:
            f.write("\n".join(lines))
    def perform_correlation_matrix(self, analysis, path, datasheet, item_names):
        items = [item_names[var] for var in analysis["variables"]]
        data = datasheet[items]
        corrs = data.corr()
        fig, ax = plt.subplots(figsize=(4+0.25*len(items), 4+0.25*len(items)))
        ax.matshow(corrs, cmap="RdYlGn", vmin=-1, vmax=1)
        fig.subplots_adjust(top=0.25*len(items)/(4+0.25*len(items)), left=4/(4+0.25*len(items)), bottom=0.02, right=0.98)
        font = {"fontsize":10}
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels(items, rotation=90, fontdict=font)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(items, fontdict=font)
        for i in range(corrs.shape[0]):
            for j in range(corrs.shape[1]):
                plt.text(j, i, str(round(corrs.iloc[i, j], 2)).replace("0.", "."), va="center", ha="center", fontdict=font)
        plt.savefig(os.path.join(path, "correlation_matrix.png"))
        plt.close()
    def variables_by_id(self):
        mapping = {}
        for coding in self.configuration["coding"]:
            for i in coding["coded_ids"]:
                mapping[i] = coding["raw_name"]
        for scale in self.configuration["scales"]:
            for i in scale["all_ids"]:
                mapping[i] = scale["name"]
        return mapping
    def compute_descriptives(self, identifier, column, reliabilities):
        lines = []
        lines.append(f"Variable: {column.name}")
        lines.append(f"Mean: {np.round(column.mean(), 2)}")
        lines.append(f"Standard deviation: {np.round(column.std(), 2)}")
        lines.append(f"Skewness: {np.round(column.skew(), 2)}")
        if identifier in reliabilities:
            lines.append(f"Alpha: {np.round(reliabilities[identifier], 2)}")
        lines.append("")
        coding = next((x for x in self.configuration["coding"] if any(column.name.startswith(q) for q in x["coded_ids"]) and column.name.endswith(x["raw_name"])), None)
        if coding != None:
            if len(column.unique()) < 20:
                lines.append("Distribution:")
                for value in np.sort(column.unique()):
                    proportion = (column == value).mean()
                    spots = ("#" * int(np.ceil(proportion * 40))).ljust(40)
                    perc = str(np.round(proportion * 100, 1)).rjust(5) + "%"
                    if type(coding["key"]) is dict:
                        value = "/".join(key for key, code in coding["key"].items() if value == code)
                    lines.append(f" {perc} | {spots} | {value}")
        return "\n".join(lines)
    def on_save(self):
        prev_last_save = self.configuration["last_save"]
        try:
            self.configuration["last_save"] = str(datetime.now())
            project_str = yaml.dump(self.configuration)
            with open(self.config_path, "w") as f:
                f.write(project_str)
            with open(os.path.join(self.output_path, "backups", os.path.basename(self.config_path) + ".backup-" + str(self.configuration["last_save"]).replace(":", "")), "w") as f:
                f.write(project_str)
        except:
            self.configuration["last_save"] = prev_last_save
            print("Error.")
            traceback.print_exc()

class MenuState(State):
    def __init__(self, parent):
        self.parent = parent
        self.options = []
        self.add_option("x", "Cancel", lambda: None)
    def clear_options(self):
        pass
    def add_option(self, hotkey, name, func):
        def on_event():
            result = func()
            if result is None:
                return self.parent
            else:
                return result
        super().add_option(hotkey, name, on_event)
    def on_after_query(self):
        self.parent.on_after_query()

class FormState(State):
    def __init__(self, parent, done):
        self.schema = []
        self.object = {}
        self.parent = parent
        self.done = done
    def add_property(self, name, ptype, query):
        self.schema.append((name, ptype, query))
    def on_help(self):
        print("Current settings:")
        print()
        for key, value in self.object.items():
            print(f"{key}: {value}")
        print()
        print("Are you satisfied with those?")
    def on_reset(self):
        self.object = {}
    def on_query(self):
        self.add_option("x", "Cancel", self.parent.open)
        self.add_option("r", "Reset", self.on_reset)
        self.add_option("y", "Confirm", lambda: self.done(self.object))
    def query(self):
        for name, ptype, query in self.schema:
            if name not in self.object:
                if type(ptype) is list:
                    value = perform_list_query(query, ptype)
                    while value is None:
                        print("Invalid response.")
                        value = perform_list_query(query, ptype)
                elif ptype == "free":
                    print(query)
                    value = input("> ")
                elif ptype == "numbers":
                    print(query)
                    value = input("> ")
                    while not all(is_int(x.strip()) for x in value.split(",")):
                        print("Invalid response. Must be a number, or a comma-separated list of numbers.")
                        value = input("> ")
                    value = [int(x.strip()) for x in value.split(",")]
                elif ptype == "items":
                    variable_ids = self.parent.variable_ids()
                    def select_items(value):
                        result = []
                        parts = value.split(",")
                        for part in parts:
                            part = part.split("-")
                            part = [p.strip() for p in part]
                            if len(part) not in [1, 2]:
                                return None
                            if any(not p in variable_ids for p in part):
                                return None
                            if len(part) == 1:
                                result.append(part[0])
                            else:
                                result.extend(sorted([i for i in variable_ids if i >= part[0] and i <= part[1]]))
                        return result
                    print(query)
                    value = input("> ")
                    while select_items(value) is None:
                        if value != "?":
                            print("Invalid response. Must be an item id (e.g. q000), item range (e.g. q000-q010), or comma-separated list of items and item ranges.")
                            print("Write '?' to see the list of possible items.")
                        else:
                            items_by_id = self.parent.variables_by_id()
                            for item in sorted(variable_ids):
                                print(f"{item} - {items_by_id[item]}")
                        value = input("> ")
                    value = select_items(value)
                elif ptype == "id":
                    id_re = re.compile("[a-z_]+")
                    print(query)
                    value = input("> ")
                    while not id_re.match(value):
                        print("Invalid response. Must consist of only alphabetical characters and underscores.")
                        value = input("> ")
                else:
                    raise Exception(f"Unrecognized ptype {ptype}")
                self.object[name] = value
                return self.open
        return super().query()



args = sys.argv
if len(args) == 1:
    state = UnloadedState()
elif len(args) == 2:
    state = load_dataset(os.path.abspath(args[1]))
else:
    print("Usage:")
    print()
    print("python statstool.py                  - prompts to load project")
    print("python statstool.py <dataset name>   - loads project and engages the stats tool")
    print()
    print("Note: if you tried using the second usage and this message appeared, maybe you accidentally included spaces in the dataset name. Try escaping the spaces and run this again.")
    exit(0)

while True:
    event = state.query()
    if event is None:
        print("Invalid option!")
    else:
        result = event()
        state.on_after_query()
        if result is not None:
            state = result