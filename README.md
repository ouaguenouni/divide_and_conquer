# A Quick Guide to Divide & Conquer Algorithms

## 1. What Are Divide & Conquer Algorithms?

Divide & Conquer algorithms are designed to break complex problems into smaller, more manageable pieces. They follow three fundamental steps: **Divide**, **Conquer**, and **Combine**.

### Breakdown of the Steps:

- **Divide**: The problem is split into smaller subproblems, each similar to the original. The key is to find a way to divide the problem while keeping the structure intact so that solving each subproblem can help with the overall solution.

- **Conquer**: Each subproblem is solved, often recursively. This involves calling the same function for the smaller subproblems that were created in the previous step.

- **Combine**: The solutions of the subproblems are merged to form the solution to the original problem. You assume the subproblems are already solved and focus on how their solutions fit together.

At each step, you're effectively answering: 
*“How can I break this problem down so that solving its parts leads directly to solving the whole?”*

## 2.Master Theorem: In-Depth Explanation of the Three Cases

The Master Theorem helps determine the time complexity of divide-and-conquer algorithms by analyzing how the work is distributed across recursive calls. Let’s examine how recursion and problem division contribute to the overall complexity in each case.

#### Case 1: When $f(n) = O(n^c)$ and $c < \log_b a$

In this case, recursion dominates the complexity. The total work is primarily determined by the number of nodes, especially at the deepest level, where the most nodes exist.

- The depth of the recursion tree is determined by how many times we can divide the problem size $n$ by $b$, which is $\log_b(n)$.
- At each level, the number of subproblems (nodes) increases by a factor of $a$. The total number of nodes at the leaf level is $a^{\log_b(n)}$, which simplifies to $n^{\log_b a}$.
- Since the work per leaf node is small, the overall complexity is dominated by the number of leaf nodes, leading to:

$$ T(n) = O(n^{\log_b a}) $$

#### Case 2: When $f(n) = O(n^c)$ and $c > \log_b a$

Here, the work done at each step grows faster than the recursive process itself, meaning that the work at individual nodes outweighs the impact of the divisions.

- The node at the root holds the largest problem size, which is $n$.
- Since the work at each step dominates, the total complexity is directly determined by the root node, leading to:

$$ T(n) = O(n^c) $$

#### Case 3: When $f(n) = O(n^c)$ and $c = \log_b a$

In this case, the work is balanced between recursion and the individual steps at each level of the recursive tree.

- The size of each subproblem decreases at the same rate as the number of nodes increases. This creates a balance, where the contributions from all levels are similar to the work done at the root.
- The overall complexity considers the work at each level ($O(n^c)$) and the number of levels in the recursion tree, which is $\log_b n$. This leads to the total complexity:

$$ T(n) = O(n^c \log_b n) $$

### Summary of the Three Cases

- **Case 1**: Recursion dominates, with the complexity driven by the number of recursive calls and the number of leaf nodes. The complexity is $O(n^{\log_b a})$.
- **Case 2**: The work at each node grows faster than the recursive steps, so the overall complexity is dominated by node-level work, resulting in $O(n^c)$.
- **Case 3**: There’s a balance between problem size reduction and the number of subproblems, leading to $O(n^c \log_b n)$, where the logarithmic factor accounts for the depth of recursion.

## 3.The Merge Sort Problem

**Input:** An array of $n$ unsorted elements.

**Output:** A sorted array with the elements arranged in non-decreasing order.

- **Intuition of the solution**: The idea behind Merge Sort is to recursively break down the array into smaller subarrays until each subarray contains a single element (which is trivially sorted). Then, these sorted subarrays are merged back together in a way that ensures the entire array remains sorted.

- **How the problem is divided**: The array is divided into two approximately equal halves. The same sorting task is applied to both halves, meaning each subproblem has the same structure as the original. Therefore, $b = 2$.

- **How many recursive calls we make**: For each division into two subarrays, two recursive calls are made—one for the left half and one for the right half. Thus, $a = 2$.

- **How we recover the solution**: After the recursive calls return the sorted subarrays, we merge the two subarrays into one sorted array. The merging process requires $O(n)$ work, as each element from the subarrays is compared and placed in the correct order. So, $f(n) = O(n)$, or $O(n^c)$ where $c = 1$.

Now, applying the Master Theorem:

$$ T(n) = aT\left(\frac{n}{b}\right) + f(n) $$

Substituting the values:

- $a = 2$
- $b = 2$
- $f(n) = O(n^1)$

We compute $\log_b a$:

$$ \log_b a = \log_2 2 = 1 $$

Since $c = 1$ and $\log_b a = 1$, this corresponds to **Case 2** of the Master Theorem, where $f(n) = O(n^c) = O(n^{log_b(a)})$.

So the total complexity is: $T(n) \in \Theta(n \log n) $

### Merge Sort Code

```python
def merge_sort(arr):
    if len(arr) > 1:
        # Step 1: Divide the array into two halves
        mid = len(arr) // 2
        left = arr[:mid]  # Left half of the array
        right = arr[mid:]  # Right half of the array

        # Step 2: Recursive calls on the left and right halves
        merge_sort(left)  # Recursive call to sort the left half
        merge_sort(right)  # Recursive call to sort the right half

        # Step 3: Merge the two sorted halves back together
        i, j, k = 0, 0, 0  # Pointers for left, right, and original arrays
        while i < len(left) and j < len(right):  # While both halves have unmerged elements
            if left[i] < right[j]:  # If left element is smaller, add it to the sorted array
                arr[k] = left[i]
                i += 1
            else:  # Otherwise, add the right element to the sorted array
                arr[k] = right[j]
                j += 1
            k += 1

        # Step 4: Merge any remaining elements from the left half
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        # Step 5: Merge any remaining elements from the right half
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
```

## 4.The Max-Min Problem

**Input:** An array of $n$ unsorted elements.

**Output:** The maximum and minimum elements from the array.

- **Intuition of the solution**: The solution recursively breaks the array into smaller parts, and each part is processed to find its maximum and minimum values. These values are then combined to find the overall maximum and minimum.

- **How the problem is divided**: The array is split into two halves, and the same operation (finding max and min) is applied to both halves. Therefore, $b = 2$.

- **How many recursive calls we make**: We make two recursive calls—one for each half of the array. Thus, $a = 2$.

- **How we recover the solution**: After each recursive call finds the max and min for its respective half, we combine the results by comparing the maxima and minima of the two halves. The work done to combine them is constant, so $f(n) = O(1)$, or $O(n^c)$ where $c = 0$.

Now, applying the Master Theorem:

$$ T(n) = aT\left(\frac{n}{b}\right) + f(n) $$

Substituting the values:

- $a = 2$
- $b = 2$
- $f(n) = O(1)$

We compute $\log_b a$:

$$ \log_b a = \log_2 2 = 1 $$

Since $f(n) = O(n^0)$ and $c = 0 < \log_b a = 1$, this corresponds to **Case 1** of the Master Theorem, where $f(n)$ grows slower than $n^{\log_b a}$.

So the total complexity is: $T(n) \in \Theta(n)$.

### Max-Min Code

```python
def find_max_min(arr, left, right):
    """
    Find the maximum and minimum values in a given array using a divide-and-conquer approach.
    
    Parameters:
    arr (list): The array of unsorted elements.
    left (int): The starting index of the array (or subarray).
    right (int): The ending index of the array (or subarray).
    
    Returns:
    tuple: A tuple containing two values:
           - max_value (int): The maximum value in the array.
           - min_value (int): The minimum value in the array.
    """
    
    # Base case for arrays with two elements: directly return the max and min.
    if right - left == 1:
        return max(arr[left], arr[right]), min(arr[left], arr[right])
    
    # Base case for arrays with one element: the only element is both max and min.
    elif right - left == 0:
        return arr[left], arr[left]

    # Step 1: Divide - Split the array into two halves.
    mid = (left + right) // 2  # Find the midpoint index
    
    # Step 2: Conquer - Recursive call to process the left half.
    max_l, min_l = find_max_min(arr, left, mid)
    
    # Step 3: Conquer - Recursive call to process the right half.
    max_r, min_r = find_max_min(arr, mid + 1, right)

    # Step 4: Combine - Compare the max and min of both halves and return the overall max and min.
    return max(max_l, max_r), min(min_l, min_r)

# Test the function with an example array
arr = [3, 2, 5, 1, 2, 7, 8, 8]
# Find the maximum and minimum values in the array
max_num, min_num = find_max_min(arr, 0, len(arr) - 1)
# Output the results
print(f"Maximum: {max_num}, Minimum: {min_num}")
```

## 5.The Exponentiation Problem

**Input:** A base value $x$ and an exponent $m$.

**Output:** The result of $x$ raised to the power of $m$, i.e., $x^m$.

- **Intuition of the solution**: The solution leverages the fact that exponentiation can be optimized by dividing the problem into smaller powers. Instead of multiplying $x$ by itself $m$ times, we reduce the problem size by recursively halving the exponent when it’s even, or decrementing it by 1 when it’s odd.

- **How the problem is divided**: When $m$ is even, we reduce the exponent by half, solving the smaller subproblem of $x^{m/2}$. When $m$ is odd, we decrement $m$ by 1 and solve for $x^{m-1}$. Therefore, the problem size is reduced by a factor of 2 each time, making $b = 2$.

- **How many recursive calls we make**: We make only one recursive call per step, regardless of whether $m$ is odd or even. Thus, $a = 1$.

- **How we recover the solution**: After recursively solving the subproblems, if $m$ is even, we multiply the result by itself ($temp \times temp$). If $m$ is odd, we multiply the result by $x$. The work done at each step is constant, so $f(n) = O(1)$, corresponding to $O(n^c)$ where $c = 0$.

Now, applying the Master Theorem:

$$ T(n) = aT\left(\frac{n}{b}\right) + f(n) $$

Substituting the values:

- $a = 1$
- $b = 2$
- $f(n) = O(1)$

We compute $\log_b a$:

$$ \log_b a = \log_2 1 = 0 $$

Since $f(n) = O(n^0)$ and $c = 0 = \log_b a = 0$, this corresponds to **Case 2** of the Master Theorem, where $f(n) = O(n^c) = O(n^{\log_b a})$.

So the total complexity is: $T(n) \in \Theta(\log m)$.

### Exponentiation Code

```python
def power(x, m):
    """
    Computes x raised to the power of m using a divide-and-conquer approach.

    Parameters:
    x (int or float): The base value.
    m (int): The exponent value (non-negative integer).

    Returns:
    int or float: The result of x raised to the power of m.
    """
    
    # Base case: any number raised to the power of 0 is 1
    if m == 0:
        return 1
    
    # If the exponent is even
    elif m % 2 == 0:
        # Recursive call to compute power for half the exponent
        temp = power(x, m // 2)
        # Combine step: square the result of x^(m//2)
        return temp * temp
    
    # If the exponent is odd
    else:
        # Recursive call to compute power for (m-1)
        temp = power(x, m - 1)
        # Combine step: multiply x with the result
        return x * temp

# Test the function
print(power(2, 5))  # Expected Result: 32
```

## 6.The Majority Element Problem

**Input:** An array $A$ of size $n$ where a majority element exists (an element that appears more than $n/2$ times).

**Output:** The majority element of the array.

- **Intuition of the solution**: The idea is to recursively divide the array into two halves, determine the majority element in each half, and then combine the results by checking which of the two potential majority elements appears more frequently in the entire array.

- **How the problem is divided**: The array is split into two halves at each recursive step, and the same task (finding the majority element) is applied to each half. Therefore, $b = 2$.

- **How many recursive calls we make**: At each step, two recursive calls are made—one for the left half and one for the right half. Thus, $a = 2$.

- **How we recover the solution**: After recursively finding the majority element for each half, we count the occurrences of both candidates (from the left and right) in the full range and return the one that appears most frequently. Counting each candidate takes linear time $O(n)$. Therefore, $f(n) = O(n)$, or $O(n^c)$ where $c = 1$.

Now, applying the Master Theorem:

$$ T(n) = aT\left(\frac{n}{b}\right) + f(n) $$

Substituting the values:

- $a = 2$
- $b = 2$
- $f(n) = O(n)$

We compute $\log_b a$:

$$ \log_b a = \log_2 2 = 1 $$

Since $f(n) = O(n^1)$ and $c = \log_b a = 1$, this corresponds to **Case 2** of the Master Theorem, where $f(n) = O(n^c) = O(n^{\log_b a})$.

So the total complexity is: $T(n) \in \Theta(n \log n)$.

### Majority Element Code

```python
def majority_element_rec(A, low, high):
    """
    Finds the majority element in the subarray A[low:high] using divide-and-conquer.
    
    Parameters:
    A (list): The array of elements.
    low (int): The starting index of the subarray.
    high (int): The ending index of the subarray.
    
    Returns:
    int: The majority element in the subarray.
    """
    
    # Base case: If the subarray has only one element, return it
    if low == high:
        return A[low]
    
    # Step 1: Divide - Find the midpoint of the current subarray
    mid = (low + high) // 2
    
    # Step 2: Conquer - Recursively find majority in left and right halves
    left_majority = majority_element_rec(A, low, mid)
    right_majority = majority_element_rec(A, mid + 1, high)
    
    # Step 3: Combine - If both halves have the same majority, return it
    if left_majority == right_majority:
        return left_majority
    
    # Count occurrences of both candidates in the entire subarray
    left_count = sum(1 for i in range(low, high + 1) if A[i] == left_majority)
    right_count = sum(1 for i in range(low, high + 1) if A[i] == right_majority)
    
    # Return the element that appears more frequently
    return left_majority if left_count > right_count else right_majority

def majority_element(A):
    """
    Finds the majority element in the array A using the recursive helper function.
    
    Parameters:
    A (list): The array of elements.
    
    Returns:
    int: The majority element in the array.
    """
    return majority_element_rec(A, 0, len(A) - 1)

# Example usage
A = [2, 2, 1, 1, 2, 2]
print(majority_element(A))  # Output: 2

```

## 7.The Median of Two Sorted Arrays Problem

**Input:** Two sorted arrays $A$ and $B$ of sizes $n$ and $m$.

**Output:** The median value of the combined, sorted array formed by merging $A$ and $B$.

- **Intuition of the solution**: The recursive approach works by selecting a partition point in $A$, and calculating the corresponding partition point in $B$ such that the left side of both partitions together forms half of the total number of elements. We then check if the partition is valid i.e.
  - The biggest element in the left side of A must be smaller than the smallest element at the right side of B.
  - The biggest element in the left side of B must be smaller than the smallest element at the right side of A.
- **How the problem is divided**: At each recursive step, we adjust the partition index in $A$ based on whether the partition is valid. This reduces the search space by half, meaning $b = 2$.
- **How many recursive calls we make**: Each recursive call narrows down the partition point by halving the search space. Thus, $a = 1$.
- **How we recover the solution**: Once the partition is valid, the median is computed by taking the maximum of the left partition and the minimum of the right partition in constant time. Therefore, $f(n) = O(1)$, or $O(n^c)$ where $c = 0$.

Now, applying the Master Theorem:

$$ T(n) = aT\left(\frac{n}{b}\right) + f(n) $$

Substituting the values:

- $a = 1$
- $b = 2$
- $f(n) = O(1)$

We compute $\log_b a$:

$$ \log_b a = \log_2 1 = 0 $$

Since $f(n) = O(n^0)$ and $c = 0 = \log_b a$, this corresponds to **Case 2** of the Master Theorem, where $f(n) = O(n^c) = O(n^{\log_b a})$.

So the total complexity is: $T(n) \in \Theta(\log \min(n, m))$.

### Find Median of Two Sorted Arrays (Recursive with Index Partitioning)

```python
def find_median_recursive(A, B, imin, imax, half_len):
    """
    Recursively finds the median of two sorted arrays A and B by computing the partition points in both arrays.
    
    Parameters:
    A (list): The first sorted array.
    B (list): The second sorted array.
    imin (int): The starting index for the binary search on array A.
    imax (int): The ending index for the binary search on array A.
    half_len (int): Half the combined length of A and B.
    
    Returns:
    float: The median of the combined sorted arrays.
    """
    
    # Base condition for invalid state
    if imin > imax:
        raise ValueError("Input arrays are not sorted correctly.")
    
    i = (imin + imax) // 2  # Partition index in A
    j = half_len - i  # Corresponding partition index in B
    
    # Case 1: i is too small, need to increase it
    if i < len(A) and B[j - 1] > A[i]:
        return find_median_recursive(A, B, i + 1, imax, half_len)
    
    # Case 2: i is too big, need to decrease it
    elif i > 0 and A[i - 1] > B[j]:
        return find_median_recursive(A, B, imin, i - 1, half_len)
    
    # Case 3: i is perfect, calculate the median
    else:
        # Max of the left partition
        if i == 0:
            max_of_left = B[j - 1]
        elif j == 0:
            max_of_left = A[i - 1]
        else:
            max_of_left = max(A[i - 1], B[j - 1])
        
        # If the total length is odd, return the max of the left partition
        if (len(A) + len(B)) % 2 == 1:
            return max_of_left
        
        # Min of the right partition
        if i == len(A):
            min_of_right = B[j]
        elif j == len(B):
            min_of_right = A[i]
        else:
            min_of_right = min(A[i], B[j])
        
        # Return the average of the two middle values for even length
        return (max_of_left + min_of_right) / 2.0


def find_median_sorted_arrays(A, B):
    """
    Initiates the recursive binary search to find the median of two sorted arrays A and B.
    
    Parameters:
    A (list): The first sorted array.
    B (list): The second sorted array.
    
    Returns:
    float: The median of the combined sorted arrays.
    """
    if len(A) > len(B):
        A, B = B, A
    
    n, m = len(A), len(B)
    return find_median_recursive(A, B, 0, n, (n + m + 1) // 2)

# Example usage
A = [1, 3, 8]
B = [7, 9, 10]
print(find_median_sorted_arrays(A, B))  # Expected Output: 7.5
```

## 8.The Skyline Problem

**Input:** A list of buildings, each represented as a triplet $(L, R, H)$, where:

  - $L$ is the x-coordinate of the left edge of the building.
  - $R$ is the x-coordinate of the right edge of the building.
  - $H$ is the height of the building.

**Output:** A list of key points representing the skyline, where each key point is a pair $(x, h)$:

  - $x$ is the x-coordinate where the skyline changes.
  - $h$ is the new height of the skyline after that point.

### Intuition of the solution:

The **Skyline Problem** can be solved using a divide-and-conquer approach. We divide the list of buildings into two halves, recursively compute the skyline for each half, and then merge the two skylines to produce the final skyline. This mirrors the merge step in the Merge Sort algorithm.

### How the problem is divided:

We recursively split the list of buildings in half at each step until each subproblem contains only one building. Thus, $b = 2$ (the problem size is halved at each step).

### How many recursive calls we make:

At each step, we divide the list of buildings into two halves, making two recursive calls—one for the left half and one for the right half. Therefore, $a = 2$.

### How we recover the solution:

After recursively computing the skyline for each half, the two skylines are merged together using the `merge_skylines` function. This merging operation involves linear work, as we iterate through both skylines and combine them in $O(n)$ time. Thus, $f(n) = O(n)$, or $O(n^c)$ where $c = 1$.

### Applying the Master Theorem:

The recurrence relation for the time complexity is:

$$ T(n) = aT\left(\frac{n}{b}\right) + f(n) $$

Substituting the values:

- $a = 2$
- $b = 2$
- $f(n) = O(n)$

We compute $\log_b a$:

$$ \log_b a = \log_2 2 = 1 $$

Since $f(n) = O(n^1)$ and $c = \log_b a = 1$, this corresponds to **Case 2** of the Master Theorem, where $f(n) = O(n^c) = O(n^{\log_b a})$.

So the total complexity is: 

$$ T(n) \in \Theta(n \log n) $$

### Code Explanation

```python
def merge_skylines(left, right):
    """
    Merges two skylines into a single skyline.
    
    Parameters:
    left (list): The skyline from the left half, represented as a list of tuples (x, h).
    right (list): The skyline from the right half, represented as a list of tuples (x, h).
    
    Returns:
    list: The merged skyline, represented as a list of tuples (x, h).
    """
    merged = []
    h1 = h2 = 0  # Heights from the left and right skylines
    i = j = 0  # Pointers for left and right skylines

    # Merge both skylines by comparing x-coordinates
    while i < len(left) and j < len(right):
        if left[i][0] < right[j][0]:
            x, h1 = left[i]  # Take the x-coordinate and height from the left skyline
            i += 1
        elif left[i][0] > right[j][0]:
            x, h2 = right[j]  # Take the x-coordinate and height from the right skyline
            j += 1
        else:
            # Same x-coordinate, take the maximum height from both
            x = left[i][0]
            h1 = left[i][1]
            h2 = right[j][1]
            i += 1
            j += 1
        
        # Compute the maximum height at the current x-coordinate
        max_h = max(h1, h2)
        # Only add the point if the height has changed
        if not merged or merged[-1][1] != max_h:
            merged.append((x, max_h))
    
    # Append the remaining points from both skylines
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged

def find_skyline(buildings, start, end):
    """
    Recursively finds the skyline for a list of buildings.
    
    Parameters:
    buildings (list): List of buildings, each represented as (L, R, H).
    start (int): The starting index for the current subproblem.
    end (int): The ending index for the current subproblem.
    
    Returns:
    list: The skyline represented as a list of tuples (x, h).
    """
    
    # Base case: If there's only one building, return its skyline
    if start == end:
        L, R, H = buildings[start]
        return [(L, H), (R, 0)]
    
    # Divide the problem into two halves
    mid = (start + end) // 2
    # Recursively find the skyline for the left and right halves
    left_skyline = find_skyline(buildings, start, mid)
    right_skyline = find_skyline(buildings, mid + 1, end)
    
    # Merge the two skylines
    return merge_skylines(left_skyline, right_skyline)

# Example usage
buildings = [(1, 3, 3), (2, 4, 4), (5, 6, 1)]
result = find_skyline(buildings, 0, len(buildings) - 1)
print(result)  # Expected Output: [(1, 3), (2, 4), (4, 0), (5, 1), (6, 0)]
```

## 9.Find First and Last Position of a Target in a Sorted Array

**Input:** A sorted array $A$ and a target value $target$.

**Output:** A tuple $(first, last)$, where:

  - $first$ is the index of the first occurrence of $target$ in $A$.
  - $last$ is the index of the last occurrence of $target$ in $A$.
  - If $target$ is not found in $A$, return $(-1, -1)$.

### Intuition of the solution:

The problem is asking for the first and last positions of a target in a sorted array. Instead of performing a linear scan, we can utilize **binary search** to efficiently find the first and last positions.

1. **Find First Position**: Perform a binary search where, upon finding the target, you continue searching in the left half of the array to find the first occurrence.
2. **Find Last Position**: Perform a binary search where, upon finding the target, you continue searching in the right half to find the last occurrence.

### How the problem is divided:

Each binary search splits the array in half at each step, reducing the search space by half. Thus, $b = 2$.

### How many recursive calls we make:

Since we are using binary search, the search space is halved at each step. This gives us a logarithmic number of recursive calls, equivalent to $O(\log n)$. Therefore, $a = 1$.

### How we recover the solution:

Both binary search operations (to find the first and last positions) take $O(\log n)$ time. The work at each step is constant, meaning $f(n) = O(1)$. Therefore, the overall complexity for both searches is $O(\log n)$.

### Total Complexity:

Since each of the two binary search functions takes $O(\log n)$ time, the total time complexity of the solution is:

$$ T(n) = O(\log n) $$

### Code Explanation:

```python
def find_last(A, target):
    """
    Finds the last occurrence of target in the sorted array A using binary search.
    
    Parameters:
    A (list): The sorted array.
    target (int): The target value to find.
    
    Returns:
    int: The index of the last occurrence of target, or -1 if not found.
    """
    low, high = 0, len(A) - 1
    last_pos = -1
    while low <= high:
        mid = (low + high) // 2
        if A[mid] == target:
            last_pos = mid
            low = mid + 1  # Continue searching in the right half
        elif A[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return last_pos

def find_first_and_last(A, target):
    """
    Finds the first and last occurrences of target in the sorted array A.
    
    Parameters:
    A (list): The sorted array.
    target (int): The target value to find.
    
    Returns:
    tuple: A tuple containing the indices of the first and last occurrences of target.
    """
    first = find_first(A, target)
    last = find_last(A, target)
    return (first, last)

# Example usage
A = [1, 2, 2, 2, 3, 4, 5]
target = 2
print(find_first_and_last(A, target))  # Expected Output: (1, 3)

```

## 10.Finding Peak Element in a 2D Array

**Input:** A 2D matrix $mat$ of size $n \times m$, where each row is sorted in any arbitrary order.

**Output:** A pair of indices $[i, j]$ where $mat[i][j]$ is a peak element. A peak element is defined as an element that is greater than or equal to its four neighbors (up, down, left, right), if they exist.

### Intuition of the solution:

The problem can be efficiently solved using **binary search** on columns. Instead of scanning all the elements, we reduce the search space by selecting the middle column, finding the global maximum in that column, and then moving either left or right based on the neighboring elements. The key observation is that if an element is not a peak, we can safely move towards its larger neighbor because there is guaranteed to be a peak in that direction.

### How the problem is divided:

At each step, we choose the middle column and find the row containing the global maximum in that column. We then reduce the search space to either the left or right half of the matrix. Thus, $b = 2$.

### How many recursive calls we make:

We make only one recursive call per step because we either move left or right depending on the neighboring elements. Therefore, $a = 1$.

### How we recover the solution:

The work done at each step is scanning all the rows in the middle column to find the global maximum, which takes linear time $O(n)$.

### Applying the Master Theorem:

The recurrence relation for the problem is:

$$ T(n, m) = T\left(n, \frac{m}{2}\right) + O(n) $$

Where:

- $a = 1$ (one recursive call),
- $b = 2$ (problem size halved at each step),
- $f(n) = O(n)$ (linear work to scan all rows in the middle column).

We compute $\log_b a$:

$$ \log_b a = \log_2 1 = 0 $$

Now, we compare $f(n)$ with $n^{\log_b a} = n^0 = 1$:

- The function $f(n) = O(n)$ grows faster than $n^{\log_b a} = n^0 = 1$, so this falls under **Case 3** of the Master Theorem.

In **Case 3**, when $f(n)$ grows faster than $n^{\log_b a}$, the total complexity is dominated by $f(n)$. Thus, the time complexity is:

$$ T(n, m) = O(n \log m) $$

### Time Complexity:

- **Rows**: We scan all rows in the middle column, which takes $O(n)$.
- **Columns**: We perform binary search on the columns, which takes $O(\log m)$.

Thus, the total time complexity is:

$$ O(n \log m) $$

### Code Explanation:

```python
def findPeakGrid(mat):
    """
    Finds a peak element in a 2D matrix using binary search on columns.
    
    Parameters:
    mat (list of list): The 2D matrix of integers.
    
    Returns:
    list: A pair of indices [i, j] representing the position of a peak element.
    """
    
    stcol = 0
    endcol = len(mat[0]) - 1  # Starting and ending indices of the column search space

    while stcol <= endcol:  # Binary search condition for columns

        midcol = stcol + (endcol - stcol) // 2  # Find the middle column
        ansrow = 0  # Initialize the row of the global peak element in the current column

        # Finding the row of the maximum element in the middle column
        for r in range(len(mat)):
            ansrow = r if mat[r][midcol] >= mat[ansrow][midcol] else ansrow

        # Check if the current element is greater than its left and right neighbors
        valid_left = midcol - 1 >= stcol and mat[ansrow][midcol - 1] > mat[ansrow][midcol]
        valid_right = midcol + 1 <= endcol and mat[ansrow][midcol + 1] > mat[ansrow][midcol]

        # If it's a peak element (larger than both neighbors)
        if not valid_left and not valid_right:
            return [ansrow, midcol]

        # If the right neighbor is greater, move the search to the right half
        elif valid_right:
            stcol = midcol + 1

        # Otherwise, move the search to the left half
        else:
            endcol = midcol - 1

    return [-1, -1]  # Return [-1, -1] if no peak is found (although this case won't happen)

# Example usage
mat = [
    [1, 4, 3, 6],
    [2, 5, 9, 7],
    [3, 6, 8, 10],
    [10, 13, 12, 14]
]
print(findPeakGrid(mat))  # Example Output: [3, 3] (peak element: 14)
```
