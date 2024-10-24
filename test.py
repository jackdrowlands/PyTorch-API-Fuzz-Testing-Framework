import pickle

# read pickle file
with open('result_parts/result_0.pkl', 'rb') as f:
    results = pickle.load(f)
    print(results)