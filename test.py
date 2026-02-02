tokens = [1,3,2]

result = []
i = 0
mid = (len(tokens)+1)//2

while i < mid and i + mid < len(tokens):
    result.append(tokens[i])
    result.append(tokens[i + mid])
    i += 1

if len(tokens) % 2 != 0:
    result.append(tokens[mid-1])

print(result)