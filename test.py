import pickle

# read pickle file
with open('result_parts/result_1.pkl', 'rb') as f:
    result = pickle.load(f)
    # print the first 1000 characters of the result
    print(result)