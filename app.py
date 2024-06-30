
import streamlit as st
import pandas as pd
import numpy as np
data = pd.read_csv('Dataset_predic_penyakitparu2_tabel.csv')
# Remove the "No" column
data = data.drop(columns=['No'])

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-count/len(target_col)) * np.log2(count/len(target_col)) for count in counts])
    return entropy

def information_gain(data, split_attribute_name, target_name="Hasil"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data[data[split_attribute_name] == vals[i]][target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

# Example usage: Calculate information gain for each attribute
attributes = data.columns[:-1]  # All columns except the target
info_gains = {attr: information_gain(data, attr) for attr in attributes}
# DecisionTreeC45 class definition
class DecisionTreeC45:
    def __init__(self, data, target_name="Hasil"):
        self.data = data
        self.target_name = target_name
        self.tree = self._build_tree(data)

    def _build_tree(self, data):
        # Check data
        if data.empty:
            return None

        target_values = data[self.target_name].unique()

        if len(target_values) == 1:
            return target_values[0]

        attributes = data.columns.drop(self.target_name)
        if attributes.empty:
            return data[self.target_name].mode()[0]

        # Menghitung information gain
        info_gains = {attr: information_gain(data, attr, self.target_name) for attr in attributes}

        if all(gain == 0 for gain in info_gains.values()):
            return data[self.target_name].mode()[0]

        # Memilih atribut dengan information gain tertinggi
        best_attr = max(info_gains, key=info_gains.get)

        # Membuat tree
        tree = {best_attr: {}}

        # Split dataset di atribut terbaik
        for attr_value in data[best_attr].unique():
            sub_data = data[data[best_attr] == attr_value].drop(columns=[best_attr])
            subtree = self._build_tree(sub_data)
            tree[best_attr][attr_value] = subtree

        return tree

    def predict(self, instance):
        tree = self.tree
        while isinstance(tree, dict):
            attr = next(iter(tree))
            if attr in instance and instance[attr] in tree[attr]:
                tree = tree[attr][instance[attr]]
            else:
                return None
        return tree

# Buat decision tree
tree = DecisionTreeC45(data)
print(tree.tree)

# Train the decision tree
tree = DecisionTreeC45(data)

# Streamlit app
st.title("Decision Tree C4.5 Predictor")

# User input
user_input = {
    'Usia': st.selectbox('Usia', ['Tua', 'Muda']),
    'Jenis_Kelamin': st.selectbox('Jenis Kelamin', ['Pria', 'Wanita']),
    'Merokok': st.selectbox('Merokok', ['Pasif', 'Aktif']),
    'Bekerja': st.selectbox('Bekerja', ['Ya', 'Tidak']),
    'Rumah_Tangga': st.selectbox('Rumah Tangga', ['Ya', 'Tidak']),
    'Aktivitas_Begadang': st.selectbox('Aktivitas Begadang', ['Ya', 'Tidak']),
    'Aktivitas_Olahraga': st.selectbox('Aktivitas Olahraga', ['Sering', 'Jarang']),
    'Asuransi': st.selectbox('Asuransi', ['Ada', 'Tidak']),
    'Penyakit_Bawaan': st.selectbox('Penyakit Bawaan', ['Tidak', 'Ada'])
}

# Prediction
if st.button("Predict"):
    prediction = tree.predict(user_input)
    st.write(f"Predicted Hasil: {prediction}")
