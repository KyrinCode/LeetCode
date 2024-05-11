# LeetCode 专题

## 数组

### [面试题 17.10. 主要元素](https://leetcode-cn.com/problems/find-majority-element-lcci/)

简单 # 2020.09.01

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int len = nums.size();
        int n = len / 2;
        map<int, int> cnt;
        for(int i=0; i<len; i++){
            cnt[nums[i]]++;
            if(cnt[nums[i]] > n){
                return nums[i];
            }
        }
        return -1;
    }
};

class Solution { // 空间复杂度O(1)
public:
    int majorityElement(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return -1;
        
        int tmp = nums[0];
        int count = 0;
        
        for(int i=0; i<len; i++){
            if(tmp == nums[i]) count++;
            else count--;
            if(count == 0) tmp = nums[i], count = 1;
        }

        if(count > 0){
            int cnt = 0;
            for(int i=0; i<len; i++){
                if(nums[i] == tmp) cnt++;
            }
            if(cnt > len/2) return tmp;
        }
        return -1;
    }
};
```

### [832. 翻转图像](https://leetcode-cn.com/problems/flipping-an-image/)

简单 # 2020.09.01

```cpp
class Solution {
public:
    vector<vector<int>> flipAndInvertImage(vector<vector<int>>& A) {
        int r = A.size();
        int c = A[0].size();
        for(int i=0; i<r; i++){
            for(int j=0; j<c/2; j++){
                int tmp = A[i][j];
                A[i][j] = 1 - A[i][c-j-1];
                A[i][c-j-1] = 1 - tmp;
            }
            if(c%2 == 1){
                A[i][c/2] = 1 - A[i][c/2];
            }
        }
        return A;
    }
};
```

### [1233. 删除子文件夹](https://leetcode-cn.com/problems/remove-sub-folders-from-the-filesystem/)

中等 # 2020.09.02

```cpp
class Solution {
public:
    vector<string> removeSubfolders(vector<string>& folder) {
        int len = folder.size();
        sort(folder.begin(), folder.end());
        vector<string> ans;
        string tmp = folder[0];
        ans.push_back(tmp);
        int tmp_size = tmp.size();
        for(int i=1; i<len; i++){
            if(tmp == folder[i].substr(0, tmp_size) && folder[i][tmp_size] == '/')
                continue;
            else ans.push_back(folder[i]), tmp = folder[i], tmp_size = tmp.size();
        }
        return ans;
    }
};
```

### [969. 煎饼排序](https://leetcode-cn.com/problems/pancake-sorting/)

中等 # 2020.09.02

```cpp
class Solution {
public:
    vector<int> pancakeSort(vector<int>& A) {
        vector<int> sorted = A;
        sort(sorted.begin(), sorted.end());
        int len = A.size();
        vector<int> ans;
        for(int i=len-1; i>=0; i--){
            int j;
            for(j=0; j<i; j++){
                if(A[j] == sorted[i]){
                    ans.push_back(j+1);
                    break;
                }
            }
            if(j == i) continue;
            for(int k=0; k<=j/2; k++){
                int tmp = A[j-k];
                A[j-k] = A[k];
                A[k] = tmp;
            }
            ans.push_back(i+1);
            for(int k=0; k<=i/2; k++){
                int tmp = A[i-k];
                A[i-k] = A[k];
                A[k] = tmp;
            }
        }
        // for(int k=0; k<len; k++){
        //     cout<<A[k];
        // }
        return ans;
    }
};
```

### [64. 最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

中等 # 远古

```cpp
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        if(m == 0)
            return 0;
        int n = grid[0].size();
        vector<vector<int>> d(m, vector<int>(n, 0));
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(i == 0 && j == 0)
                    d[i][j] = grid[i][j];
                else{
                    d[i][j] = min(i-1>=0?d[i-1][j]:INT_MAX, j-1>=0?d[i][j-1]:INT_MAX) + grid[i][j];
                }
            }
        }
        return d[m-1][n-1];
    }
};
```

### [78. 子集](https://leetcode-cn.com/problems/subsets/)

中等 # 远古

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& S) { // 递归
        vector<vector<int> > res;
        vector<int> out;
        sort(S.begin(), S.end());
        getSubsets(S,0,out,res);
        return res;
     }
     void getSubsets(vector<int> &S, int pos, vector<int> &out, vector<vector<int> > &res) {
         res.push_back(out);
         for (int i = pos; i < S.size(); i++) {
             out.push_back(S[i]);
             getSubsets(S,i+1,out,res);
             out.pop_back();
         }
     }
};

class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) { // 位运算
        int size = nums.size();
        vector<vector<int> > ans;
        for(int i=0; i<(1<<size); i++){
            vector<int> s;
            for(int j=0; j<size; j++){
                if(i&(1<<j))
                    s.push_back(nums[j]);
            }
            ans.push_back(s);
        }
        return ans;
    }
};
```

### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

困难 # 远古

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        vector<int> level = height;
        vector<int> ans(n, 0);
        int i = 0;
        while(i < n-1){
            int j;
            for(j=i+1; j<n; j++){
                if(height[j] >= height[i]){
                    for(int k=i+1; k<j; k++){
                        level[k] = height[i];
                    }
                    i = j;
                    break;
                }
            }
            if(j == n)
                i++;
        }
        i = n-1;
        while(i > 0){
            int j;
            for(j=i-1; j>=0; j--){
                if(height[j] >= height[i]){
                    for(int k=i-1; k>j; k--){
                        level[k] = max(level[k], height[i]);
                    }
                    i = j;
                    break;
                }
            }
            if(j == -1)
                i--;
        }
        // for(int i=0; i<n; i++){
        //     cout<<level[i]<<" ";
        // }
        // cout<<endl;
        for(int i=0; i<n; i++){
            ans[i] = level[i] - height[i];
        }
        return accumulate(ans.begin(), ans.end(), 0);
    }
};
```

### [830. 较大分组的位置](https://leetcode-cn.com/problems/positions-of-large-groups/)

简单 # 2021.01.05

```cpp
class Solution {
public:
    vector<vector<int>> largeGroupPositions(string s) {
        vector<vector<int>> ans;
        int len = s.size();
        s += s[len-1]+1; // 补一个和最后一位不一样的字符
        int start = 0, end;
        char tmp = s[0];
        for(int i=1; i<len+1; i++){
            if(s[i] != tmp){
                end = i-1;
                if(end - start > 1)
                    ans.push_back({start, end});
                start = i;
                tmp = s[i];
            }
        }
        return ans;
    }
};
```

### [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

中等 # 2021.01.08

```cpp
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k % n;
        int cnt = gcd(k, n); // 最大公约数，cnt个圈，各圈从0、1、2开始
        for(int i=0; i<cnt; i++){
            int idx = i;
            int last = nums[i];
            do {
                int new_idx = (idx + k) % n;
                swap(nums[new_idx], last);
                idx = new_idx;
            } while (i != idx);
        }
    }
};
```



### [1018. 可被 5 整除的二进制前缀](https://leetcode-cn.com/problems/binary-prefix-divisible-by-5/)

简单 # 2020.01.14

```cpp
class Solution {
public:
    vector<bool> prefixesDivBy5(vector<int>& A) {
        int len = A.size();
        int x = 0;
        vector<bool> ans(len, false);
        for(int i=0; i<len; i++){
            x = (x*2 + A[i])%5;
            if(x%5 == 0) ans[i] = true;
        }
        return ans;
    }
};
```

### [1232. 缀点成线](https://leetcode-cn.com/problems/check-if-it-is-a-straight-line/)

简单 # 2021.01.17

```cpp
class Solution {
public:
    bool checkStraightLine(vector<vector<int>>& coordinates) {
        int len = coordinates.size(), k;
        for(int i=1; i<len; i++){
            coordinates[i][0] = coordinates[i][0] - coordinates[0][0];
            coordinates[i][1] = coordinates[i][1] - coordinates[0][1];
            if (i > 1 && coordinates[i][0]*coordinates[1][1]!=coordinates[i][1]*coordinates[1][0] ){
                return false;
            }
        }
        return true;
    }
};
```

### [628. 三个数的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/)

简单 # 2021.01.20

```cpp
class Solution {
public:
    int maximumProduct(vector<int>& nums) {
        sort(nums.begin(),nums.end());
        int len = nums.size();
        // 返回三个最大正数乘积与两个最小负数和最大正数的乘积的最大值
        return max(nums[len-1] * nums[len-2] * nums[len-3],nums[0] * nums[1] * nums[len-1]);
    }
};
```

### [989. 数组形式的整数加法](https://leetcode-cn.com/problems/add-to-array-form-of-integer/)

简单 # 2021.01.22

```cpp
class Solution {
public:
    vector<int> addToArrayForm(vector<int>& A, int K) {
        vector<int> k, ans;
        while(K > 0){
            k.push_back(K % 10);
            K /= 10;
        }
        k.push_back(0); // 都多补个0
        reverse(A.begin(), A.end());
        A.push_back(0); // 都多补个0
        int lenA = A.size();
        int lenK = k.size();
        int maxlen = max(lenA, lenK);
        int C = 0;
        for(int i=0; i<maxlen; i++){
            int tmp = (i<lenA ? A[i] : 0) + (i<lenK ? k[i] : 0) + C;
            if(tmp > 9) C = 1;
            else C = 0;
            ans.push_back(tmp%10);
        }
        if(ans[maxlen-1] == 0) ans.pop_back();
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```

### [724. 寻找数组的中心索引](https://leetcode-cn.com/problems/find-pivot-index/)

简单 # 2021.01.28

```cpp
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        int len = nums.size();
        int total = accumulate(nums.begin(), nums.end(), 0);
        int sum = 0;
        for(int i=0; i<len; i++){
            if (2 * sum + nums[i] == total) {
                return i;
            }
            sum += nums[i];
        }
        return -1;
    }
};
```

### [888. 公平的糖果棒交换](https://leetcode-cn.com/problems/fair-candy-swap/)

简单 # 2021.02.01

```cpp
class Solution {
public:
    vector<int> fairCandySwap(vector<int>& A, vector<int>& B) {
        int alen=A.size(), blen=B.size(), asum=0, bsum=0;
        for(int i=0; i<alen; i++){
            asum += A[i];
        }
        for(int j=0; j<blen; j++){
            bsum += B[j];
        }
        int diff = asum - bsum;
        sort(A.begin(), A.end());
        sort(B.begin(), B.end());
        int i=0, j=0;
        while(i<alen && j<blen){
            int tmp = A[i]-B[j];
            if(tmp==diff/2){
                return vector<int>{A[i], B[j]};
            } else if(tmp>diff/2) j++;
            else i++;
        }
        if(i<alen){
            for(; j<blen; j++) if(A[i]-B[j]==diff/2) return vector<int>{A[i], B[j]};
        }
        if(j<blen){
            for(; i<alen; i++) if(A[i]-B[j]==diff/2) return vector<int>{A[i], B[j]};
        }
        return vector<int>{A[i], B[j]};
    }
};
```

### [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

简单 # 2021.02.04

```cpp
class Solution {
public:
    double findMaxAverage(vector<int>& nums, int k) {
        double sum = 0, ans;
        int len = nums.size();
        for(int i=0; i<k; i++){
            sum += nums[i];
        }
        ans = sum;
        for(int i=k; i<len; i++){
            sum -= nums[i-k];
            sum += nums[i];
            ans = max(ans, sum);
        }
        return ans/k;
    }
};
```

### [665. 非递减数列](https://leetcode-cn.com/problems/non-decreasing-array/)

简单 # 2021.02.07

```cpp
class Solution {
public:
    bool checkPossibility(vector<int>& nums) { // 遇到的第一次递减，要么都变成大，要么都变成小
        int len = nums.size();
        pair<int, int> tmp;
        for(int i=0; i<len-1; i++){
            if(nums[i] > nums[i+1]){
                tmp.first = i;
                tmp.second = nums[i];
                nums[i] = nums[i+1];
                break;
            }
        }
        int i = 0;
        for(; i<len-1; i++){
            if(nums[i] > nums[i+1]) break;
        }
        if(i == len-1) return true;
        nums[tmp.first] = tmp.second;
        nums[tmp.first+1] = tmp.second;
        i = 0;
        for(; i<len-1; i++){
            if(nums[i] > nums[i+1]) return false;
        }
        return true;
    }
};
```

### [448. 找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

简单 # 2021.02.13

```cpp
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) { // 原地修改
        int len = nums.size();
        vector<int> ans;
        for(int i=0; i<len; i++){
            nums[(nums[i]-1)%len] += len;
        }
        for(int i=0; i<len; i++){
            if(nums[i] <= len) ans.push_back(i+1);
        }
        return ans;
    }
};
```

### [485. 最大连续1的个数](https://leetcode-cn.com/problems/max-consecutive-ones/)

简单 # 2021.02.15

```cpp
class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int maxCount = 0, cnt = 0;
        int len = nums.size();
        for(int i=0; i<len; i++){
            if(nums[i] == 1){
                cnt++;
            } else {
                maxCount = max(maxCount, cnt);
                cnt = 0;
            }
        }
        maxCount = max(maxCount, cnt);
        return maxCount;
    }
};
```

### [561. 数组拆分 I](https://leetcode-cn.com/problems/array-partition-i/)

简单 # 2021.02.16

```cpp
class Solution {
public:
    int arrayPairSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int sum = 0;
        int n = nums.size();
        for(int i=0; i<n; i+=2){
            sum += nums[i];
        }
        return sum;
    }
};
```

### [566. 重塑矩阵](https://leetcode-cn.com/problems/reshape-the-matrix/)

简单 # 2021.02.17

```cpp
class Solution {
public:
    vector<vector<int>> matrixReshape(vector<vector<int>>& nums, int r, int c) {
        int R = nums.size();
        int C = nums[0].size();
        if(R*C != r*c){
            return nums;
        }
        vector<vector<int>> ans(r, vector<int>(c));
        for(int i=0; i<R; i++){
            for(int j=0; j<C; j++){
                int x = i * C + j;
                ans[x/c][x%c] = nums[i][j];
            }
        }
        return ans;
    }
};
```

### [995. K 连续位的最小翻转次数](https://leetcode-cn.com/problems/minimum-number-of-k-consecutive-bit-flips/)

困难 # 2021.02.18

```cpp
class Solution {
public:
    int minKBitFlips(vector<int>& A, int K) {
        int len = A.size();
        vector<int> diff(len+1); // 差分数组：从idx开始对后面所有都进行操作
        int ans = 0, cnt = 0;
        for(int i=0; i<len; i++){
            cnt += diff[i];
            if((A[i]+cnt)%2 == 0){
                if(i+K > len) return -1;
                ans++;
                cnt++; // 这里加相当于 diff[i]++;cnt+=diff[i]
                diff[i+K]--; // 这里减
            }
        }
        return ans;
    }
};
```

### [697. 数组的度](https://leetcode-cn.com/problems/degree-of-an-array/)

简单 # 2021.02.20

```cpp
class Solution {
public:
    int findShortestSubArray(vector<int>& nums) {
        unordered_map<int, vector<int>> m; // 数组存出现次数、第一次出现的位置、最后一次出现的位置
        int len = nums.size();
        for(int i=0; i<len; i++){
            if(m.count(nums[i])){
                m[nums[i]][0]++;
                m[nums[i]][2] = i;
            } else {
                m[nums[i]] = {1, i, i};
            }
        }
        int val = -1; int maxCnt = 0; int minLen = 0;
        for(auto x: m){
            if(x.second[0] > maxCnt || (x.second[0] == maxCnt && x.second[2]-x.second[1]+1 < minLen)){
                val = x.first;
                maxCnt = x.second[0];
                minLen = x.second[2]-x.second[1]+1;
            }
        }
        int l=0, r=len-1;
        for(; l<len; l++){
            if(nums[l] == val) break;
        }
        for(; r>=0; r--){
            if(nums[r] == val) break;
        }
        return r-l+1;
    }
};
```

### [766. 托普利茨矩阵](https://leetcode-cn.com/problems/toeplitz-matrix/)

简单 # 2021.02.22

```cpp
class Solution {
public:
    bool isToeplitzMatrix(vector<vector<int>>& matrix) {
        int r = matrix.size();
        int c = matrix[0].size();
        for(int i=1; i<r; i++){
            for(int j=1; j<c; j++){
                if(matrix[i][j] != matrix[i-1][j-1]) return false;
            }
        }
        return true;
    }
};
```

### [867. 转置矩阵](https://leetcode-cn.com/problems/transpose-matrix/)

简单 # 2021.02.25

```cpp
class Solution {
public:
    vector<vector<int>> transpose(vector<vector<int>>& matrix) {
        int r = matrix.size();
        int c = matrix[0].size();
        vector<vector<int>> ans(c, vector<int>(r));
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                ans[j][i] = matrix[i][j];
            }
        }
        return ans;
    }
};
```

### [896. 单调数列](https://leetcode-cn.com/problems/monotonic-array/)

简单 # 2021.02.28

```cpp
class Solution {
public:
    bool isMonotonic(vector<int> &A) {
        bool inc = true, dec = true;
        int len = A.size();
        for(int i=0; i<len-1; i++){
            if(A[i] > A[i+1]){
                inc = false;
            }
            if(A[i] < A[i + 1]){
                dec = false;
            }
        }
        return inc || dec;
    }
};
```

### [303. 区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)

简单 # 2021.03.01

```cpp
class NumArray {
public:
    vector<int> accu;
    NumArray(vector<int>& nums) {
        int len = nums.size();
        accu.push_back(0);
        for(int i=0; i<len; i++){
            accu.push_back(accu[i]+nums[i]);
        }
    }
    
    int sumRange(int i, int j) {
        return accu[j+1]-accu[i];
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(i,j);
 */
```

### [304. 二维区域和检索 - 矩阵不可变](https://leetcode-cn.com/problems/range-sum-query-2d-immutable/)

中等 # 2021.03.02

```cpp
class NumMatrix {
public:
    vector<vector<int>> accu;
    NumMatrix(vector<vector<int>>& matrix) {
        int r = matrix.size();
        if(r == 0) return;
        int c = matrix[0].size();
        accu.resize(r);
        for(int i=0; i<r; i++){
            accu[i].resize(c);
            for(int j=0; j<c; j++){
                accu[i][j] = (i>0?accu[i-1][j]:0) + (j>0?accu[i][j-1]:0) - (i>0&&j>0?accu[i-1][j-1]:0) + matrix[i][j];
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        return accu[row2][col2] - (col1>0?accu[row2][col1-1]:0) - (row1>0?accu[row1-1][col2]:0) + (row1>0&&col1>0?accu[row1-1][col1-1]:0);
    }
};

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix* obj = new NumMatrix(matrix);
 * int param_1 = obj->sumRegion(row1,col1,row2,col2);
 */
```

### [59. 螺旋矩阵 II](https://leetcode-cn.com/problems/spiral-matrix-ii/)

中等 # 2021.03.16

```cpp
class Solution {
public:
    int circle(vector<vector<int>>& ans, int i_s, int i_e, int start){
        if(i_s == i_e){
            ans[i_s][i_s] = start; start++;
            return start;
        } else {
            for(int i=i_s; i<i_e; i++){
                ans[i_s][i] = start; start++;
            }
            for(int i=i_s; i<i_e; i++){
                ans[i][i_e] = start; start++;
            }
            for(int i=i_e; i>i_s; i--){
                ans[i_e][i] = start; start++;
            }
            for(int i=i_e; i>i_s; i--){
                ans[i][i_s] = start; start++;
            }
            return start;
        }
    }
    vector<vector<int>> generateMatrix(int n) {
        vector<vector<int>> ans(n, vector<int>(n));
        int start = 1, i = 0;
        while(start <= n*n){
            start = circle(ans, i, n-1-i, start);
            i++;
        }
        return ans;
    }
};
```

### [73. 矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)

中等 # 2021.03.21

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        vector<int> row(m), col(n);
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(!matrix[i][j]){
                    row[i] = col[j] = true;
                }
            }
        }
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++) {
                if(row[i] || col[j]){
                    matrix[i][j] = 0;
                }
            }
        }
    }
};
```

### [456. 132 模式](https://leetcode-cn.com/problems/132-pattern/)

中等 # 2021.03.24

```cpp
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        int n = nums.size();
        if(n < 3) return false;
        int left_min = nums[0]; // 左侧维护最小值
        multiset<int> right_all; // // 右侧所有元素排序
        for(int i=2; i<n; i++){
            right_all.insert(nums[i]);
        }
        for(int i=1; i<n-1; i++){
            if(left_min < nums[i]){
                auto it = right_all.upper_bound(left_min);
                if(it != right_all.end() && *it < nums[i]){
                    return true;
                }
            }
            left_min = min(left_min, nums[i]);
            right_all.erase(right_all.find(nums[i+1]));
        }
        return false;
    }
};
```

### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

简单 # 2021.04.19

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        for(int i=0; i<nums.size(); i++){
            if(nums[i] == val){
                nums[i] = nums[nums.size()-1];
                nums.pop_back();
                i--;
            }
        }
        return nums.size();
    }
};
```

### [363. 矩形区域不超过 K 的最大数值和](https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/)

困难 # 2021.04.22

```cpp
class Solution {
public:
    int maxSumSubmatrix(vector<vector<int>>& matrix, int k) {
        int r = matrix.size();
        int c = matrix[0].size();
        int s[101][101];
        // vector<vector<int>> s(r+1, vector<int>(c+1, 0));
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                s[i+1][j+1] = s[i][j+1] + s[i+1][j] - s[i][j] + matrix[i][j];
            }
        }
        int ans = INT_MIN;
        for(int i=1; i<=r; i++){
            for(int j=1; j<=c; j++){
                for(int ii=0; ii<i; ii++){
                    for(int jj=0; jj<j; jj++){
                        int sum = s[i][j] - s[ii][j] - s[i][jj] + s[ii][jj];
                        if(sum <= k) ans = max(ans, sum);
                    }
                }
            }
        }
        return ans;
    }
};
```

## 字符串

### [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

简单 # 远古

```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int n = strs.size();
        if(n == 0)
            return "";
        int minlen = strs[0].size();
        for(int i=0; i<n; i++){
            minlen = min(minlen, int(strs[i].size()));
        }
        int p;
        for(p=0; p<minlen; p++){
            char c = strs[0][p];
            bool flag = false;
            for(int i=0; i<n; i++){
                if(strs[i][p] != c){
                    flag = true;
                    break;
                }
            }
            if(flag)
                break;
        }
        string ans = "";
        for(int i=0; i<p; i++){
            ans += strs[0][i];
        }
        return ans;
    }
};
```

### [面试题 01.02. 判定是否互为字符重排](https://leetcode-cn.com/problems/check-permutation-lcci/)

简单 # 2020.09.06

```cpp
class Solution {
public:
    bool CheckPermutation(string s1, string s2) {
        if(s1.size() == s2.size()){
            sort(s1.begin(), s1.end());
            sort(s2.begin(), s2.end());
            return s1 == s2;
        }
        return false;
    }
};
```

### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

中等 # 2020.09.06

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        int l = s.size();
        vector<int> v(l, 0);
        string ans = "";
        for(int i=0; i<l; i++){
            if(i == 0) v[i] = 1, ans = s[0];
            else{
                v[i] = v[i-1];
                for(int j=0; j<i; j++){
                    if(s[j] == s[i] && isP(s, j, i)){
                        int newlen = i - j + 1;
                        if(newlen > v[i-1]){
                            v[i] = newlen;
                            ans = s.substr(j, newlen);
                        }
                        break;
                    }
                }
            }
        }
        return ans;
    }
    bool isP(string &s, int front, int back){
        int mid = (front + back) / 2;
        for(int i=0; i<=mid-front; i++){
            if(s[front+i] != s[back-i]){
                return false;
            }
        }
        return true;
    }
};
```

### [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

中等 # 2020.09.07

```cpp
class Solution {
public:
    int myAtoi(string str) {
        stringstream ss;
        ss<<str;
        ss>>str;
        int len = str.size();
        if(len == 0) return 0;
        int sign = 1, i = 0;
        long ans;
        if(str[0] == '+') i++;
        else if(str[0] == '-') sign = -1, i++;
        ans = 0;
        int cnt = 0;
        for(; i<len; i++){
            cnt++;
            if(isdigit(str[i])){
                ans = ans * 10 + (str[i] - '0');
                if(sign * ans > INT_MAX)
                    return INT_MAX;
                else if(sign * ans < INT_MIN)
                    return INT_MIN;
            }
            else{
                if(cnt == 1)
                    return 0;
                break;
            }
        }
        return sign * ans;
    }
};
```

### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

难度：中等 # 远古

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1：**

```
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```


**示例 2：**
```
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```
**示例 3：**
```
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```
**示例 4：**
```
输入: s = ""
输出: 0
```

**提示：**

+ `0 <= s.length <= 5 * 10^4`
+ `s` 由英文字母、数字、符号和空格组成

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int size = s.size();
        int i=0, j, k, ans=0;
        for(j=0; j<size; j++){
            for(k=i; k<j; k++){
                if(s[k] == s[j]){
                    i = k + 1;
                    break;
                }
            }
            if(j-i+1 > ans){
                ans = j-i+1;
            }
        }
        return ans;
    }
};
```

### [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

难度：中等 # 2020.09.08

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

**示例：**
```
输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```
**说明：**

+ 所有输入均为小写字母。
+ 不考虑答案输出的顺序。

```cpp
class Solution {
public:
    static bool cmp(pair<string, int> a, pair<string, int> b){
        return a.first < b.first;
    }
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        int len = strs.size();
        vector<vector<string>> ans;
        if(len == 0) return ans;
        vector<pair<string, int>> vp(len, pair("", 0));
        for(int i=0; i<len; i++){
            vp[i].first = strs[i];
            sort(vp[i].first.begin(), vp[i].first.end());
            vp[i].second = i;
        }
        sort(vp.begin(), vp.end(), cmp);
        vector<string> vs;
        vs.push_back(strs[vp[0].second]);
        for(int i=1; i<len; i++){
            if(vp[i].first != vp[i-1].first){
                ans.push_back(vs);
                vs.clear();
            }
            vs.push_back(strs[vp[i].second]);
        }
        ans.push_back(vs);
        return ans;
    }
};
```

### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

难度：困难 # 2020.09.10

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

注意：如果 `s` 中存在这样的子串，我们保证它是唯一的答案。

**示例 1：**
```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```
**示例 2：**
```
输入：s = "a", t = "a"
输出："a"
```
**提示：**

+ `1 <= s.length, t.length <= 10^5`
+ `s` 和 `t` 由英文字母组成

**进阶：**你能设计一个在 o(n) 时间内解决此问题的算法吗？

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        int l = 0, r = 0;
        int slen = s.size(), tlen = t.size();
        map<char, int> m, mwin;
        int valid = 0, start = 0, ansl = INT_MAX;
        for(int i=0; i<tlen; i++){
            m[t[i]]++;
        }
        while(r < slen){
            char c = s[r++];
            if(m.count(c)){
                mwin[c]++;
                if(mwin[c] == m[c])
                    valid++;
            }
            while(valid == m.size()){
                if(ansl > r - l){
                    ansl = r - l;
                    start = l;
                    // cout<<"start: "<<start<<" ansl: "<<ansl<<endl;
                    // cout<<"from "<<s[start]<<" to "<<s[start+ansl-1]<<endl;
                }
                char c2 = s[l++];
                if(m.count(c2)){
                    if(mwin[c2] == m[c2])
                        valid--;
                    mwin[c2]--;
                }
            }
        }
        return ansl == INT_MAX ? "" : s.substr(start, ansl);
    }
};
```

### [1689. 十-二进制数的最少数目](https://leetcode-cn.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/)

难度：中等 # 2021.03.14

如果一个十进制数字不含任何前导零，且每一位上的数字不是 `0` 就是 `1` ，那么该数字就是一个 **十-二进制数**。例如，`101` 和 `1100` 都是 **十-二进制数**，而 `112` 和 `3001` 不是。

给你一个表示十进制整数的字符串 `n` ，返回和为 `n` 的 **十-二进制数** 的最少数目。

**示例 1：**
```
输入：n = "32"
输出：3
解释：10 + 11 + 11 = 32
```
**示例 2：**
```
输入：n = "82734"
输出：8
```
**示例 3：**
```
输入：n = "27346209830709182346"
输出：9
```

**提示：**

+ `1 <= n.length <= 10^5`
+ `n` 仅由数字组成
+ `n` 不含任何前导零并总是表示正整数

```cpp
class Solution {
public:
    int minPartitions(string n) {
        int len = n.size();
        int ans = 0;
        for(int i=0; i<len; i++){
            ans = max(ans, n[i]-'0');
        }
        return ans;
    }
};
```

### [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

难度：简单 # 2021.04.20

实现 `strStr()` 函数。

给你两个字符串 `haystack` 和 `needle` ，请你在 `haystack` 字符串中找出 `needle` 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  `-1` 。

**说明：**

当 `needle` 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 `needle` 是空字符串时我们应当返回 0 。这与 C 语言的 `strstr()` 以及 Java 的 `indexOf()` 定义相符。

**示例 1：**
```
输入：haystack = "hello", needle = "ll"
输出：2
```
**示例 2：**
```
输入：haystack = "aaaaa", needle = "bba"
输出：-1
```
**示例 3：**
```
输入：haystack = "", needle = ""
输出：0
```
**提示：**

+ `0 <= haystack.length, needle.length <= 5 * 10^4`
+ `haystack` 和 `needle` 仅由小写英文字符组成

```cpp
class Solution {
public:
    int strStr(string haystack, string needle) {
        int n = haystack.size();
        int m = needle.size();
        if(m == 0)
            return 0;
        int i;
        for(i=0; i<=n-m; i++){
            int j;
            for(j=i; j-i<m; j++){
                if(haystack[j] != needle[j-i]){
                    break;
                }
            }
            if(j-i == m)
                return i;
        }
        return -1;
    }
};
```



## 动态规划

### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

难度：中等 # 2020.09.13

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**示例 1：**
```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
```
**示例 2：**
```
输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。
```

**提示：**

+ `0 <= nums.length <= 100`
+ `0 <= nums[i] <= 400`

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        vector<int> v(len, 0);
        for(int i=0; i<len; i++){
            if(i == 0) v[i] = nums[0];
            else if(i == 1) v[i] = max(v[0], nums[1]);
            else{
                v[i] = max(v[i-1], v[i-2] + nums[i]);
            }
        }
        return v[len-1];
    }
};
```

### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

难度：中等 # 2020.09.14

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例：**
```
输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```
**进阶：**

如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int len = nums.size();
        int pre = 0;
        int ans = INT_MIN;
        for(int i=0; i<len; i++){
            if(i == 0) pre = nums[0];
            else pre = max(pre + nums[i], nums[i]);
            ans = max(ans, pre);
        }
        return ans;
    }
};
```

### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

难度：中等 # 远古

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

**示例 1：**

![robot_maze](./images/robot_maze.png)
```
输入：m = 3, n = 7
输出：28
```
**示例 2：**
```
输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
```
**示例 3：**
```
输入：m = 7, n = 3
输出：28
```
示例 4：
```
输入：m = 3, n = 3
输出：6
```

**提示：**

+ `1 <= m, n <= 100`
+ 题目数据保证答案小于等于 `2 * 10^9`

```cpp
class Solution {
public:
    int dp[101][101] = {0};
    int uniquePaths(int m, int n) {
        if(m==1 || n==1){
            return 1;
        }
        if(dp[m][n] > 0){
            return dp[m][n];
        }
        else{
            dp[m][n] = uniquePaths(m-1, n) + uniquePaths(m, n-1);
            return uniquePaths(m-1, n) + uniquePaths(m, n-1);
        }
    }
};
```

### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

难度：中等 # 2020.09.16

给定一个**非**空字符串 s 和一个包含非空单词的列表 *wordDict*，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

**说明：**

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。
**示例 1：**
```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```
**示例 2：**
```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```
**示例 3：**
```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

```cpp
class Solution {
public:
    // 将字典按长度降序排序
    bool static cmp(string a, string b){
        return a.size() > b.size();
    }
    bool recursion(string s, int& slen_origin, vector<string>& wordDict, int& dlen, vector<bool>& v){
        int slen = s.size();
        // 访问过，返回false
        if(!v[slen_origin - slen]) return false;
        // 空串，返回true
        if(slen == 0) return true;
        // 遍历字典
        for(int i=0; i<dlen; i++){
            string w = wordDict[i];
            int wlen = w.size();
            int j;
            // 比较s开头与当前遍历的字典项
            for(j=0; j<wlen; j++){
                if(s[j] != w[j]) break;
            }
            if(j == wlen){
                if(recursion(s.substr(wlen, slen - wlen), slen_origin, wordDict, dlen, v)) return true;
                else v[slen_origin - slen + wlen] = false;
            }
        }
        return false;
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        int slen_origin = s.size();
        int dlen = wordDict.size();
        // 记录从index开始到s末尾的子串是否满足
        vector<bool> v(slen_origin, true);
        sort(wordDict.begin(), wordDict.end(), cmp);
        return recursion(s, slen_origin, wordDict, dlen, v);
    }
};
```

### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

难度：中等 # 远古

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**示例 1：**
```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```
**示例 2：**
```
输入：nums = [0,1,0,3,2,3]
输出：4
```
**示例 3：**
```
输入：nums = [7,7,7,7,7,7,7]
输出：1
```
**提示：**

+ `1 <= nums.length <= 2500`
+ `-104 <= nums[i] <= 104`

**进阶：**

+ 你可以设计时间复杂度为 `O(n2)` 的解决方案吗？
+ 你能将算法的时间复杂度降低到 `O(n log(n))` 吗?

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        if(n == 0)
            return 0;
        int d[n];
        for(int i=0; i<n; i++){
            d[i] = 1;
        }
        for(int i=1; i<n; i++){
            for(int j=0; j<i; j++){
                if(nums[i] > nums[j]){
                    d[i] = max(d[i], d[j]+1);
                }
            }
        }
        int maxlen = 1;
        for(int i=0; i<n; i++){
            maxlen = max(maxlen, d[i]);
        }
        return maxlen;
    }
};
```

### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

难度：困难 # 2020.09.17

给你两个单词 `word1` 和 `word2`，请你计算出将 `word1` 转换成 `word2` 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

+ 插入一个字符
+ 删除一个字符
+ 替换一个字符

**示例 1：**
```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```
**示例 2：**
```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```
**提示：**

+ `0 <= word1.length, word2.length <= 500`
+ `word1` 和 `word2` 由小写英文字母组成

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        int len1 = word1.size();
        int len2 = word2.size();
        if(len1 == 0) return len2;
        if(len2 == 0) return len1;
        vector<vector<int>> dp(len1+1, vector<int>(len2+1, 0));
        // word1前i子串和word2前j子串间最少操作数
        for(int i=0; i<=len1; i++){
            for(int j=0; j<=len2; j++){
                if(i + j == 0) continue;
                else if(i * j == 0) dp[i][j] = max(i, j);
                else{
                    // a增加1或b增加1或a修改1（最后一位相同不必修改）
                    if(word1[i-1] == word2[j-1]) dp[i][j] = min(min(dp[i-1][j]+1, dp[i][j-1]+1), dp[i-1][j-1]);
                    else dp[i][j] = min(min(dp[i-1][j]+1, dp[i][j-1]+1), dp[i-1][j-1]+1);
                }
            }
        }
        return dp[len1][len2];
    }
};
```

### [174. 地下城游戏](https://leetcode-cn.com/problems/dungeon-game/)

难度：困难 # 2020.09.18

一些恶魔抓住了公主（**P**）并将她关在了地下城的右下角。地下城是由 M x N 个房间组成的二维网格。我们英勇的骑士（**K**）最初被安置在左上角的房间里，他必须穿过地下城并通过对抗恶魔来拯救公主。

骑士的初始健康点数为一个正整数。如果他的健康点数在某一时刻降至 0 或以下，他会立即死亡。

有些房间由恶魔守卫，因此骑士在进入这些房间时会失去健康点数（若房间里的值为*负整数*，则表示骑士将损失健康点数）；其他房间要么是空的（房间里的值为 *0*），要么包含增加骑士健康点数的魔法球（若房间里的值为*正整数*，则表示骑士将增加健康点数）。

为了尽快到达公主，骑士决定每次只向右或向下移动一步。

**编写一个函数来计算确保骑士能够拯救到公主所需的最低初始健康点数。**

例如，考虑到如下布局的地下城，如果骑士遵循最佳路径 `右 -> 右 -> 下 -> 下`，则骑士的初始健康点数至少为 **7**。

| -2 (K) | -3   | 3      |
| ------ | ---- | ------ |
| -5     | -10  | 1      |
| 10     | 30   | -5 (P) |

**说明:**

- 骑士的健康点数没有上限。
- 任何房间都可能对骑士的健康点数造成威胁，也可能增加骑士的健康点数，包括骑士进入的左上角房间以及公主被监禁的右下角房间。

```cpp
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int r = dungeon.size(), c = dungeon[0].size();
        //从右下角往左上角，所需要的最低初始健康点数
        vector<vector<int>> dp(r + 1, vector<int>(c + 1, INT_MAX));
        dp[r][c - 1] = dp[r - 1][c] = 1;
        for (int i = r - 1; i >= 0; i--) {
            for (int j = c - 1; j >= 0; j--) {
                dp[i][j] = max(min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 1);
            }
        }
        return dp[0][0];
    }
};
```

### [509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

难度：简单 # 2021.01.04

**斐波那契数**，通常用 `F(n)` 表示，形成的序列称为 **斐波那契数列** 。该数列由 `0` 和 `1` 开始，后面的每一项数字都是前面两项数字的和。也就是：
```
F(0) = 0，F(1) = 1
F(n) = F(n - 1) + F(n - 2)，其中 n > 1
```
给你 `n` ，请计算 `F(n)` 。

**示例 1：**
```
输入：2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1
```
**示例 2：**
```
输入：3
输出：2
解释：F(3) = F(2) + F(1) = 1 + 1 = 2
```
**示例 3：**
```
输入：4
输出：3
解释：F(4) = F(3) + F(2) = 2 + 1 = 3
```

**提示：**

+ 0 <= n <= 30

```cpp
class Solution {
public:
    int fib(int n) {
        int arr[2] = {0, 1};
        for(int i=2; i<=n; i++){
            arr[i & 1] = arr[0] + arr[1];
        }
        return arr[n & 1];
    }
};
```

### [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

难度：困难 # 2021.01.09

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 **两笔** 交易。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1：**
```
输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
```
**示例 2：**
```
输入：prices = [1,2,3,4,5]
输出：4
解释：在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。   
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```
**示例 3：**
```
输入：prices = [7,6,4,3,1] 
输出：0 
解释：在这个情况下, 没有交易完成, 所以最大利润为 0。
```
**示例 4：**
```
输入：prices = [1]
输出：0
```
**提示：**

+ `1 <= prices.length <= 105`
+ `0 <= prices[i] <= 105`

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len == 0) return 0;
        vector<vector<int>> dp(len, vector<int>(5, 0)); // 分成5个阶段
        // dp[0][0] = 0;
        dp[0][1] = -prices[0]; // 第1次买入后手中现金
        // dp[0][2] = 0; // 第1次卖出初始化收益为0
        dp[0][3] = -prices[0]; // 第2次买入后手中现金
        // dp[0][4] = 0; // 第2次卖出初始化收益为0
        for(int i = 1; i < len; i++){
            dp[i][0] = dp[i - 1][0]; // 保持没有操作的状态
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]); // 保持第1次买入或在当天第1次买入
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i]); // 保持第1次卖出或在当天第1次卖出
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i]); // 保持第2次买入或在当天第2次买入
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i]); // 保持第2次卖出或在当天第2次卖出
        }
        return dp[len - 1][4];
    }
};
```

### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

难度：简单 # 2021.01.09

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

注意：你不能在买入股票前卖出股票。

**示例 1：**
```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```
**示例 2：**
```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```
```cpp
class Solution { // dp
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len == 0) return 0;
        vector<vector<int>> dp(len, vector<int>(3, 0)); // 3个阶段
        // dp[0][0] = 0;
        dp[0][1] = -prices[0];
        // dp[0][2] = 0;
        for(int i=1; i<len; i++){
            dp[i][0] = dp[i-1][0];
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i]);
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] + prices[i]);
        }
        return dp[len-1][2];
    }
};

class Solution {
public:
    int maxProfit(vector<int>& prices) { // 单调栈
        stack<int> s;
        int len = prices.size();
        int bottom, ans = 0; // 记录栈底买入值，和历史买卖最大值
        for(int i=0; i<len; i++){
            if(s.empty()){
                s.push(prices[i]);
                bottom = prices[i];
            }
            else if(prices[i] > s.top()){
                s.push(prices[i]);
                ans = max(ans, prices[i]-bottom);
            }
            else{
                while(!s.empty() && s.top() >= prices[i]){
                    s.pop();
                }
                if(s.empty()) bottom = prices[i];
                s.push(prices[i]);
            }
        }
        return ans;
    }
};
```

### [674. 最长连续递增序列](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

难度：简单 # 2021.01.24

给定一个未经排序的整数数组，找到最长且 **连续递增的子序列**，并返回该序列的长度。

**连续递增的子序列** 可以由两个下标 `l` 和 `r`（`l < r`）确定，如果对于每个 `l <= i < r`，都有 `nums[i] < nums[i + 1]` ，那么子序列 `[nums[l], nums[l + 1], ..., nums[r - 1], nums[r]]` 就是连续递增子序列。

**示例 1：**
```
输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
```
**示例 2：**
```
输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。
```

**提示：**

+ `0 <= nums.length <= 104`
+ `-109 <= nums[i] <= 109`

```cpp
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        vector<int> dp(len, 1);
        int ans = 1;
        for(int i=1; i<len; i++){
            if(nums[i] > nums[i-1]){
                dp[i] = dp[i-1]+1;
                ans = max(ans, dp[i]);
            }
        }
        return ans;
    }
};
```

### [978. 最长湍流子数组](https://leetcode-cn.com/problems/longest-turbulent-subarray/)

难度：中等 # 2021.02.08

当 `A` 的子数组 `A[i], A[i+1], ..., A[j]` 满足下列条件时，我们称其为湍流子数组：

若 `i <= k < j`，当 `k` 为奇数时， `A[k] > A[k+1]`，且当 `k` 为偶数时，`A[k] < A[k+1]`；
或 若 `i <= k < j`，当 `k` 为偶数时，`A[k] > A[k+1]` ，且当 `k` 为奇数时， `A[k] < A[k+1]`。
也就是说，如果比较符号在子数组中的每个相邻元素对之间翻转，则该子数组是湍流子数组。

返回 `A` 的最大湍流子数组的长度。

**示例 1：**
```
输入：[9,4,2,10,7,8,8,1,9]
输出：5
解释：(A[1] > A[2] < A[3] > A[4] < A[5])
```
**示例 2：**
```
输入：[4,8,12,16]
输出：2
```
**示例 3：**
```
输入：[100]
输出：1
```

**提示：**

1. `1 <= A.length <= 40000`
2. `0 <= A[i] <= 10^9`

```cpp
class Solution {
public:
    int maxTurbulenceSize(vector<int>& arr) {
        int len = arr.size();
        vector<int> x(len, 0);
        vector<int> dp1(len, 0);
        vector<int> dp2(len, 0);
        int ans = 0;
        for(int i=0; i<len-1; i++){
            if((i%2==1 && arr[i]>arr[i+1]) || (i%2==0 && arr[i]<arr[i+1])) x[i] = 1;
        }
        for(int i=0; i<len; i++){
            if(x[i]==1){
                if(i==0) dp1[i] = 1;
                else dp1[i] = dp1[i-1] + 1;
                ans = max(ans, dp1[i]);
            }
        }
        for(int i=0; i<len-1; i++){
            if((i%2==0 && arr[i]>arr[i+1]) || (i%2==1 && arr[i]<arr[i+1])) x[i] = 1;
            else x[i] = 0;
        }
        for(int i=0; i<len; i++){
            if(x[i]==1){
                if(i==0) dp2[i] = 1;
                else dp2[i] = dp2[i-1] + 1;
                ans = max(ans, dp2[i]);
            }
        }
        return ans+1;
    }
};
```

### [338. 比特位计数](https://leetcode-cn.com/problems/counting-bits/)

难度：中等 # 2021.03.03

给定一个非负整数 **num**。对于 **0 ≤ i ≤ num** 范围中的每个数字 **i** ，计算其二进制数中的 1 的数目并将它们作为数组返回。

**示例 1：**
```
输入: 2
输出: [0,1,1]
```
**示例 2：**
```
输入: 5
输出: [0,1,1,2,1,2]
```
**进阶：**

+ 给出时间复杂度为 **O(n\*sizeof(integer))** 的解答非常容易。但你可以在线性时间 **O(n)** 内用一趟扫描做到吗？
+ 要求算法的空间复杂度为 **O(n)**。
+ 你能进一步完善解法吗？要求在 C++ 或任何其他语言中不使用任何内置函数（如 C++ 中的 `__builtin_popcount`）来执行此操作。

```cpp
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> ans(num+1);
        for(int i=1; i<=num; i++){
            if(i==1) ans[i] = 1;
            else {
                ans[i] = ans[i - pow(2, int(log(i)/log(2)))] + 1;
            }
        }
        return ans;
    }
};
```

### [354. 俄罗斯套娃信封问题](https://leetcode-cn.com/problems/russian-doll-envelopes/)

难度：困难 # 2020.03.04

给定一些标记了宽度和高度的信封，宽度和高度以整数对形式 `(w, h)` 出现。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算最多能有多少个信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

**说明：**
不允许旋转信封。

**示例：**
```
输入: envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出: 3 
解释: 最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
```

```cpp
class Solution {
public:
    bool static cmp(vector<int> a, vector<int> b){
        if(a[0] == b[0]) return a[1] < b[1];
        return a[0] < b[0];
    }
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int len = envelopes.size();
        if(len == 0) return 0;
        sort(envelopes.begin(), envelopes.end(), cmp);
        vector<int> dp(len, 1);
        for(int i=1; i<len; i++){
            for(int j=0; j<i; j++){
                if(envelopes[j][0] < envelopes[i][0] && envelopes[j][1] < envelopes[i][1]){
                    dp[i] = max(dp[i], dp[j]+1);
                }
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```

### [132. 分割回文串 II](https://leetcode-cn.com/problems/palindrome-partitioning-ii/)

难度：困难 # 2021.03.08

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文。

返回符合要求的 **最少分割次数** 。

**示例 1：**
```
输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
```
**示例 2：**
```
输入：s = "a"
输出：0
```
**示例 3：**
```
输入：s = "ab"
输出：1
```

**提示：**

+ `1 <= s.length <= 2000`
+ `s` 仅由小写英文字母组成

```cpp
class Solution {
public:
    bool check(int l, int r, string& s, vector<vector<bool>>& v, vector<vector<bool>>& is_palindrome){
        if(l > r) return true;
        if(v[l][r] != true){
            is_palindrome[l][r] = (s[l] == s[r]) && check(l+1, r-1, s, v, is_palindrome);
            v[l][r] = true;
        }
        return is_palindrome[l][r];
    }
    int minCut(string s) {
        int n = s.size();
        vector<vector<bool>> is_palindrome(n, vector<bool>(n, true));
        vector<vector<bool>> v(n, vector<bool>(n, false));
        for(int i=0; i<n; i++){
            v[i][i] = true;
        }
        vector<int> dp(n, INT_MAX); // 截止第i个位置需要的最少分割次数
        for(int i=0; i<n; i++){
            if(check(0, i, s, v, is_palindrome)){
                dp[i] = 0;
            } else {
                for(int j=0; j<i; j++){
                    if(check(j+1, i, s, v, is_palindrome)){
                        dp[i] = min(dp[i], dp[j] + 1);
                    }
                }
            }
        }
        return dp[n-1];
    }
};
```

### [115. 不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)

难度：困难 # 2021.03.17

给定一个字符串 `s` 和一个字符串 `t` ，计算在 `s` 的子序列中 `t` 出现的个数。

字符串的一个 **子序列** 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，`"ACE"` 是 `"ABCDE"` 的一个子序列，而 `"AEC"` 不是）

题目数据保证答案符合 32 位带符号整数范围。

**示例 1：**
```
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
(上箭头符号 ^ 表示选取的字母)
rabbbit
^^^^ ^^
rabbbit
^^ ^^^^
rabbbit
^^^ ^^^
```
**示例 2：**
```
输入：s = "babgbag", t = "bag"
输出：5
解释：
如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
(上箭头符号 ^ 表示选取的字母)
babgbag
^^ ^
babgbag
^^    ^
babgbag
^    ^^
babgbag
  ^  ^^
babgbag
    ^^^
```
**提示：**

+ `0 <= s.length, t.length <= 1000`
+ `s` 和 `t` 由英文字母组成

```cpp
class Solution {
public:
    int dp(int a, int b, string& s, string& t, int& ls, int& lt, vector<vector<int> >& d){
        if(ls - a < lt - b)
            return 0;
        if(d[a][b] != -1)
            return d[a][b];
        if(s[a] != t[b]){
            d[a][b] = dp(a+1, b, s, t, ls, lt, d);
            return d[a][b];
        }
        else{
            d[a][b] = dp(a+1, b+1, s, t, ls, lt, d) + dp(a+1, b, s, t, ls, lt, d);
            return d[a][b];
        }
    }
    int numDistinct(string s, string t) {
        int ls = s.size();
        int lt = t.size();
        vector<vector<int> > d(ls+1, vector<int>(lt+1, -1));
        for(int i=0; i<=ls; i++){
            d[i][lt] = 1;
        }
        return dp(0, 0, s, t, ls, lt, d);
    }
};
```

### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

难度：中等 # 2021.04.15

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下** ，能够偷窃到的最高金额。

**示例 1：**
```
输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
```
**示例 2：**
```
输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
```
**示例 3：**
```
输入：nums = [0]
输出：0
```

**提示：**

+ `1 <= nums.length <= 100`
+ `0 <= nums[i] <= 1000`

```cpp
class Solution {
public:
    int dp(vector<int>& nums, int l, int r, vector<vector<int>>& v){
        if(l > r) return 0;
        if(v[l][r] != -1) return v[l][r];
        v[l][r] = max(nums[l] + dp(nums, l+2, r, v), dp(nums, l+1, r, v));
        return v[l][r];
    }
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        vector<vector<int>> v(len, vector<int>(len, -1));
        for(int i=0; i<len; i++){
            v[i][i] = nums[i];
        }
        return max(nums[0] + dp(nums, 2, len-2, v), dp(nums, 1, len-1, v));
    }
};
```

### [87. 扰乱字符串](https://leetcode-cn.com/problems/scramble-string/)

难度：困难 # 2021.04.16

使用下面描述的算法可以扰乱字符串 `s` 得到字符串 `t` ：

1. 如果字符串的长度为 1 ，算法停止

2. 如果字符串的长度 > 1 ，执行下述步骤：

   + 在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 `s` ，则可以将其分成两个子字符串 `x` 和 `y` ，且满足 `s = x + y` 。

   + **随机** 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，`s` 可能是 `s = x + y` 或者 `s = y + x` 。

   + 在 `x` 和 `y` 这两个子字符串上继续从步骤 1 开始递归执行此算法。

给你两个 长度相等 的字符串 `s1` 和 `s2`，判断 `s2` 是否是 `s1` 的扰乱字符串。如果是，返回 `true` ；否则，返回 `false` 。


**示例 1：**
```
输入：s1 = "great", s2 = "rgeat"
输出：true
解释：s1 上可能发生的一种情形是：
"great" --> "gr/eat" // 在一个随机下标处分割得到两个子字符串
"gr/eat" --> "gr/eat" // 随机决定：「保持这两个子字符串的顺序不变」
"gr/eat" --> "g/r / e/at" // 在子字符串上递归执行此算法。两个子字符串分别在随机下标处进行一轮分割
"g/r / e/at" --> "r/g / e/at" // 随机决定：第一组「交换两个子字符串」，第二组「保持这两个子字符串的顺序不变」
"r/g / e/at" --> "r/g / e/ a/t" // 继续递归执行此算法，将 "at" 分割得到 "a/t"
"r/g / e/ a/t" --> "r/g / e/ a/t" // 随机决定：「保持这两个子字符串的顺序不变」
算法终止，结果字符串和 s2 相同，都是 "rgeat"
这是一种能够扰乱 s1 得到 s2 的情形，可以认为 s2 是 s1 的扰乱字符串，返回 true
```
**示例 2：**
```
输入：s1 = "abcde", s2 = "caebd"
输出：false
```
**示例 3：**
```
输入：s1 = "a", s2 = "a"
输出：true
```

**提示：**

+ `s1.length == s2.length`
+ `1 <= s1.length <= 30`
+ `s1` 和 `s2` 由小写英文字母组成

```cpp
class Solution {
private:
    // 记忆化搜索存储状态的数组
    // -1 表示 false，1 表示 true，0 表示未计算
    int v[30][30][31];
    string s1, s2;

public:
    bool checkIfSimilar(int i1, int i2, int length) {
        string tmp1 = s1.substr(i1, length), tmp2 = s2.substr(i2, length);
        sort(tmp1.begin(), tmp1.end());
        sort(tmp2.begin(), tmp2.end());
        return tmp1 == tmp2;
    }

    // 第一个字符串从 i1 开始，第二个字符串从 i2 开始，子串的长度为 length，是否和谐
    bool dfs(int i1, int i2, int length) {
        if (v[i1][i2][length]) {
            return v[i1][i2][length] == 1;
        }

        // 判断两个子串是否相等
        if (s1.substr(i1, length) == s2.substr(i2, length)) {
            v[i1][i2][length] = 1;
            return true;
        }

        // 判断是否存在字符 c 在两个子串中出现的次数不同
        if (!checkIfSimilar(i1, i2, length)) {
            v[i1][i2][length] = -1;
            return false;
        }
        
        // 枚举分割位置
        for (int i = 1; i < length; ++i) {
            // 不交换的情况
            if (dfs(i1, i2, i) && dfs(i1 + i, i2 + i, length - i)) {
                v[i1][i2][length] = 1;
                return true;
            }
            // 交换的情况
            if (dfs(i1, i2 + length - i, i) && dfs(i1 + i, i2, length - i)) {
                v[i1][i2][length] = 1;
                return true;
            }
        }

        v[i1][i2][length] = -1;
        return false;
    }

    bool isScramble(string s1, string s2) {
        memset(v, 0, sizeof(v));
        this->s1 = s1;
        this->s2 = s2;
        return dfs(0, 0, s1.size());
    }
};
```

### [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

难度：中等 # 2021.04.21

一条包含字母 `A-Z` 的消息通过以下映射进行了 **编码** ：
```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```
要 **解码** 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，`"11106"` 可以映射为：

+ `"AAJF"` ，将消息分组为 `(1 1 10 6)`
+ `"KJF"` ，将消息分组为 `(11 10 6)`
注意，消息不能分组为  `(1 11 06)` ，因为 `"06"` 不能映射为 `"F"` ，这是由于 `"6"` 和 `"06"` 在映射中并不等价。

给你一个只含数字的 **非空** 字符串 `s` ，请计算并返回 **解码** 方法的 **总数** 。

题目数据保证答案肯定是一个 **32 位** 的整数。

**示例 1：**
```
输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
```
**示例 2：**
```
输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
```
**示例 3：**
```
输入：s = "0"
输出：0
解释：没有字符映射到以 0 开头的数字。
含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。
```
**示例 4：**
```
输入：s = "06"
输出：0
解释："06" 不能映射到 "F" ，因为字符串含有前导 0（"6" 和 "06" 在映射中并不等价）。
```

**提示：**

+ `1 <= s.length <= 100`
+ `s` 只包含数字，并且可能包含前导零。

```cpp
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        vector<int> f(n+1); // f(i)表示前i个字符数能组成的编码数量
        f[0] = 1;
        for(int i=1; i<=n; i++){
            if(s[i-1] != '0'){ // 不考虑两位组合
                f[i] += f[i-1];
            }
            if(i > 1 && s[i-2] != '0' && ((s[i-2] - '0') * 10 + (s[i-1] - '0') <= 26)){ // 当前两位的组合可以编码时
                f[i] += f[i-2];
            }
        }
        return f[n];
    }
};
```

### [368. 最大整除子集](https://leetcode-cn.com/problems/largest-divisible-subset/)

 难度：中等 # 2021.04.23

给你一个由 **无重复** 正整数组成的集合 `nums` ，请你找出并返回其中最大的整除子集 `answer` ，子集中每一元素对 `(answer[i], answer[j])` 都应当满足：
+ `answer[i] % answer[j] == 0` ，或
+ `answer[j] % answer[i] == 0`
如果存在多个有效解子集，返回其中任何一个均可。

**示例 1：**
```
输入：nums = [1,2,3]
输出：[1,2]
解释：[1,3] 也会被视为正确答案。
```
**示例 2：**
```
输入：nums = [1,2,4,8]
输出：[1,2,4,8]
```
**提示：**

+ `1 <= nums.length <= 1000`
+ `1 <= nums[i] <= 2 * 10^9`
+ `nums` 中的所有整数 **互不相同**

```cpp
class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        int len = nums.size();
        sort(nums.begin(), nums.end());

        // 第 1 步：动态规划找出最大子集的个数、最大子集中的最大整数
        vector<int> dp(len, 1);
        int maxSize = 1;
        int maxVal = dp[0];
        for(int i=1; i<len; i++){
            for(int j=0; j<i; j++){
                // 题目中说「没有重复元素」很重要
                if(nums[i] % nums[j] == 0){
                    dp[i] = max(dp[i], dp[j]+1);
                }
            }
            if(dp[i] > maxSize){
                maxSize = dp[i];
                maxVal = nums[i];
            }
        }
        // 第 2 步：倒推获得最大子集
        vector<int> res;
        if(maxSize == 1){
            res.push_back(nums[0]);
            return res;
        }
        for(int i=len-1; i>=0 && maxSize>0; i--){
            if(dp[i] == maxSize && maxVal % nums[i] == 0){
                res.push_back(nums[i]);
                maxVal = nums[i];
                maxSize--;
            }
        }
        return res;
    }
};
```

### [377. 组合总和 Ⅳ](https://leetcode-cn.com/problems/combination-sum-iv/)

难度：中等 # 2021.04.24

给你一个由 不同 整数组成的数组 `nums` ，和一个目标整数 `target` 。请你从 `nums` 中找出并返回总和为 `target` 的元素组合的个数。

题目数据保证答案符合 32 位整数范围。

**示例 1：**
```
输入：nums = [1,2,3], target = 4
输出：7
解释：
所有可能的组合为：
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
请注意，顺序不同的序列被视作不同的组合。
```
**示例 2：**
```
输入：nums = [9], target = 3
输出：0
```

**提示：**

+ `1 <= nums.length <= 200`
+ `1 <= nums[i] <= 1000`
+ `nums` 中的所有元素 **互不相同**
+ `1 <= target <= 1000`

**进阶：**如果给定的数组中含有负数会发生什么？问题会产生何种变化？如果允许负数出现，需要向题目中添加哪些限制条件？

```cpp
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<int> dp(target+1, 0);
        dp[0] = 1;
        for(int i=1; i<=target; i++){
            for(int& num : nums) {
                if(num <= i && dp[i - num] < INT_MAX - dp[i]){ // 中间有的值可能会超过整数范围
                    dp[i] += dp[i-num];
                }
            }
        }
        return dp[target];
    }
};
```

### [264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

难度：中等 # 2021.05.07

给你一个整数 `n` ，请你找出并返回第 `n` 个 **丑数** 。

丑数 就是只包含质因数 `2`、`3` 和/或 `5` 的正整数。

**示例 1：**
```
输入：n = 10
输出：12
解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
```
**示例 2：**
```
输入：n = 1
输出：1
解释：1 通常被视为丑数。
```

**提示：**

+ `1 <= n <= 1690`

```cpp
class Solution {
public:
    int nthUglyNumber(int n) {
        vector<int> dp(n + 1);
        dp[1] = 1;
        int p2 = 1, p3 = 1, p5 = 1; // p存的是表中的位置
        for (int i = 2; i <= n; i++) {
            int num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
            dp[i] = min(min(num2, num3), num5);
            if (dp[i] == num2) {
                p2++;
            }
            if (dp[i] == num3) {
                p3++;
            }
            if (dp[i] == num5) {
                p5++;
            }
        }
        return dp[n];
    }
};
```



## 数学

### [892. 三维形体的表面积](https://leetcode-cn.com/problems/surface-area-of-3d-shapes/)

难度：简单 # 2020.09.21

给你一个 `n * n` 的网格 `grid` ，上面放置着一些 `1 x 1 x 1` 的正方体。

每个值 `v = grid[i][j]` 表示 v 个正方体叠放在对应单元格 `(i, j)` 上。

放置好正方体后，任何直接相邻的正方体都会互相粘在一起，形成一些不规则的三维形体。

请你返回最终这些形体的总表面积。

注意：每个形体的底面也需要计入表面积中。

**示例 1：**
![tmp-grid1](./images/tmp-grid1.jpg)

```
输入：grid = [[2]]
输出：10
```
**示例 2：**
![tmp-grid2](./images/tmp-grid2.jpg)

```
输入：grid = [[1,2],[3,4]]
输出：34
```
**示例 3：**
![tmp-grid3](./images/tmp-grid3.jpg)

```
输入：grid = [[1,0],[0,2]]
输出：16
```
**示例 4：**
![tmp-grid4](./images/tmp-grid4.jpg)

```
输入：grid = [[1,1,1],[1,0,1],[1,1,1]]
输出：32
```
**示例 5：**
![tmp-grid5](./images/tmp-grid5.jpg)

```
输入：grid = [[2,2,2],[2,1,2],[2,2,2]]
输出：46
```

**提示：**

+ `n == grid.length`
+ `n == grid[i].length`
+ `1 <= n <= 50`
+ `0 <= grid[i][j] <= 50`

```cpp
class Solution {
public:
    int surfaceArea(vector<vector<int>>& grid) {
        int r = grid.size();
        if(r == 0) return 0;
        int c = grid[0].size();
        int ans = 0;
        int dx[4] = {0, 1, 0, -1};
        int dy[4] = {-1, 0, 1, 0};
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                int pixel = 0;
                if(grid[i][j] > 0){
                    pixel += 2;
                    for(int k=0; k<4; k++){
                        int x = i + dx[k];
                        int y = j + dy[k];
                        if(x >= 0 && x < c && y >= 0 && y < r){
                            pixel += grid[i][j] > grid[x][y] ? grid[i][j] - grid[x][y] : 0;
                        }
                        else pixel += grid[i][j];
                    }
                }
                ans += pixel;
            }
        }
        return ans;
    }
};
```

### [537. 复数乘法](https://leetcode-cn.com/problems/complex-number-multiplication/)

难度：中等 # 2020.09.21

给定两个表示复数的字符串。

返回表示它们乘积的字符串。注意，根据定义 i^2 = -1 。

**示例 1：**
```
输入: "1+1i", "1+1i"
输出: "0+2i"
解释: (1 + i) * (1 + i) = 1 + i2 + 2 * i = 2i ，你需要将它转换为 0+2i 的形式。
```
**示例 2：**
```
输入: "1+-1i", "1+-1i"
输出: "0+-2i"
解释: (1 - i) * (1 - i) = 1 + i2 - 2 * i = -2i ，你需要将它转换为 0+-2i 的形式。
```
**注意：**

1. 输入字符串不包含额外的空格。
2. 输入字符串将以 **a+bi** 的形式给出，其中整数 **a** 和 **b** 的范围均在 [-100, 100] 之间。**输出也应当符合这种形式**。

```cpp
class Solution {
public:
    string complexNumberMultiply(string a, string b) {
        int pos_plus = a.find('+');
        int a_first = atoi(a.substr(0, pos_plus).c_str());
        int a_second = atoi(a.substr(pos_plus+1, a.size()-pos_plus-2).c_str());
        pos_plus = b.find('+');
        int b_first = atoi(b.substr(0, pos_plus).c_str());
        int b_second = atoi(b.substr(pos_plus+1, b.size()-pos_plus-2).c_str());
        int ans_first = a_first * b_first - a_second * b_second;
        int ans_second = a_first * b_second + a_second * b_first;
        return to_string(ans_first) + "+" + to_string(ans_second) + "i";
    }
};
```

### [1447. 最简分数](https://leetcode-cn.com/problems/simplified-fractions/)

难度：中等 # 2020.09.24

给你一个整数 n ，请你返回所有 0 到 1 之间（不包括 0 和 1）满足分母小于等于  n 的 **最简** 分数 。分数可以以 **任意** 顺序返回。

**示例 1：**
```
输入：n = 2
输出：["1/2"]
解释："1/2" 是唯一一个分母小于等于 2 的最简分数。
```
**示例 2：**
```
输入：n = 3
输出：["1/2","1/3","2/3"]
```
**示例 3：**
```
输入：n = 4
输出：["1/2","1/3","1/4","2/3","3/4"]
解释："2/4" 不是最简分数，因为它可以化简为 "1/2" 。
```
**示例 4：**
```
输入：n = 1
输出：[]
```
**提示：**

+ `1 <= n <= 100`

```cpp
class Solution {
public:
    // 辗转相除法
    bool isSim(int a, int b){
        int big = max(a, b);
        int small = min(a, b);
        int t = big % small;
        while(t){
            big = small;
            small = t;
            t = big % small;
        }
        if(small == 1) return true;
        return false;
    }
    vector<string> simplifiedFractions(int n) {
        // cout<<isSim(4, 9)<<endl;
        vector<string> ans;
        for(int i=2; i<=n; i++){
            for(int j=1; j<i; j++){
                if(isSim(j, i)) ans.push_back(to_string(j)+"/"+to_string(i));
            }
        }
        return ans;
    }
};
```

### [877. 石子游戏](https://leetcode-cn.com/problems/stone-game/)

难度：中等 # 2020.09.24

亚历克斯和李用几堆石子在做游戏。偶数堆石子**排成一行**，每堆都有正整数颗石子 `piles[i]` 。

游戏以谁手中的石子最多来决出胜负。石子的总数是奇数，所以没有平局。

亚历克斯和李轮流进行，亚历克斯先开始。 每回合，玩家从行的开始或结束处取走整堆石头。 这种情况一直持续到没有更多的石子堆为止，此时手中石子最多的玩家获胜。

假设亚历克斯和李都发挥出最佳水平，当亚历克斯赢得比赛时返回 `true` ，当李赢得比赛时返回 `false`。

**示例：**
```
输入：[5,3,4,5]
输出：true
解释：
亚历克斯先开始，只能拿前 5 颗或后 5 颗石子 。
假设他取了前 5 颗，这一行就变成了 [3,4,5] 。
如果李拿走前 3 颗，那么剩下的是 [4,5]，亚历克斯拿走后 5 颗赢得 10 分。
如果李拿走后 5 颗，那么剩下的是 [3,4]，亚历克斯拿走后 4 颗赢得 9 分。
这表明，取前 5 颗石子对亚历克斯来说是一个胜利的举动，所以我们返回 true 。
```
**提示：**

+ `2 <= piles.length <= 500`
+ `piles.length` 是偶数。
+ `1 <= piles[i] <= 500`
+ `sum(piles)` 是奇数。

```cpp
class Solution {
public:
    bool stoneGame(vector<int>& piles) {
        return true;
    }
};

class Solution { // 超时
public:
    int maxnum(vector<int>& piles, int s, int e, vector<vector<int>>& dp){
        if(dp[s][e]) return dp[s][e];
        if(s + 1 == e){
            dp[s][e] = max(piles[s], piles[e]);
            return dp[s][e];
        }
        int a = piles[s] + maxnum(piles, s+2, e, dp);
        int b = piles[s] + maxnum(piles, s+1, e-1, dp);
        int c = piles[e] + maxnum(piles, s+1, e-1, dp);
        int d = piles[e] + maxnum(piles, s, e-2, dp);
        dp[s][e] = max(max(a, b), max(c, d));
        return dp[s][e];
    }
    bool stoneGame(vector<int>& piles) {
        int len = piles.size();
        int total = 0;
        for(int i=0; i<len; i++){
            total += piles[i];
        }
        vector<vector<int>> dp(total, vector<int>(total, 0));
        int getmax = maxnum(piles, 0, len-1, dp);
        // cout<<getmax<<endl;
        if(getmax > total / 2) return true;
        return false;
    }
};
```

### [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)

难度：困难 # 远古

给出集合 `[1,2,3,...,n]`，其所有元素共有 `n!` 种排列。

按大小顺序列出所有排列情况，并一一标记，当 `n = 3` 时, 所有排列如下：

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

给定 `n` 和 `k`，返回第 `k` 个排列。

**示例 1：**
```
输入：n = 3, k = 3
输出："213"
```
**示例 2：**
```
输入：n = 4, k = 9
输出："2314"
```
**示例 3：**
```
输入：n = 3, k = 1
输出："123"
```

**提示：**

+ `1 <= n <= 9`
+ `1 <= k <= n!`

```cpp
class Solution {
public:
    // 数组v记录用没用过当前数字，返回当前第k大数字
    int kthmax(vector<int>& v, int& n, int k){
        int cnt = 0;
        for(int i=0; i<n; i++){
            if(v[i] == 0) cnt++;
            if(cnt == k){
                v[i] = 1;
                return i+1;
            }
        }
        return 0;
    }
    // 将剩余数字降序返回
    string ans_end(vector<int>& v, int& n){
        string ans = "";
        for(int i=n-1; i>=0; i--) if(v[i] == 0) ans += to_string(i+1);
        return ans;
    }
    string getPermutation(int n, int k) {
        // 阶乘
        vector<int> factorial(n+1, 0);
        factorial[1] = 1;
        for(int i=2; i<=n; i++){
            factorial[i] = factorial[i-1] * i;
        }
        vector<int> v(n, 0);
        int N = n, K = k;
        string ans = "";
        // 从前到后一位一位解决的思想
        while(N > 1){
            int quotient = K / factorial[N-1];
            K = K % factorial[N-1];
            N--;
            // cout<<"quotient: "<<quotient<<endl;
            // cout<<"K: "<<K<<endl;
            // cout<<"N: "<<N<<endl;
            // cout<<"k: "<<quotient + (K>0?1:0)<<endl;
            ans += to_string(kthmax(v, n, quotient + (K>0?1:0)));
            if(K == 0) break; // 余数为0说明剩余数字降序列出即可
        }
        ans += ans_end(v, n);
        return ans;
    }
};
```

### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

难度：中等 # 远古

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例 1：**

![addtwonumber1](./images/addtwonumber1.jpg)

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
```
**示例 2：**
```
输入：l1 = [0], l2 = [0]
输出：[0]
```
**示例 3：**
```
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

**提示：**

+ 每个链表中的节点数在范围 `[1, 100]` 内
+ `0 <= Node.val <= 9`
+ 题目数据保证列表表示的数字不含前导零

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *headnode = new ListNode(0);
        ListNode *p = headnode;
        int val = 0;
        while(val || l1 || l2)
        {
            val = val + (l1?l1->val:0) + (l2?l2->val:0);
            p->next = new ListNode(val % 10);
            p = p->next;
            val = val / 10;
            l1 = l1?l1->next:nullptr;
            l2 = l2?l2->next:nullptr;
        }
        ListNode *ans = headnode->next;
        delete headnode;
        return ans;
        
    }
};
```

### [233. 数字 1 的个数](https://leetcode-cn.com/problems/number-of-digit-one/)

难度：困难 # 2020.09.25

给定一个整数 `n`，计算所有小于等于 `n` 的非负整数中数字 `1` 出现的个数。

**示例 1：**

```
输入：n = 13
输出：6
```
**示例 2：**
```
输入：n = 0
输出：0
```
**提示：**

+ `0 <= n <= 2 * 10^9`

```cpp
class Solution {
public:
    int countDigitOne(int n) {
        int cnt = 0;
        // 按照每一位，例如个位每10有1排第1，十位每100有10排第10-19
        for(int i=1; pow(10, i-1)<=n; i++){
            int x = n / pow(10, i); // cout<<"x: "<<x<<endl;
            long long y = n % long(pow(10, i)); // cout<<"y: "<<y<<endl;
            cnt += x * pow(10, i-1);
            cnt += (y>=2*pow(10, i-1)-1?pow(10, i-1):(y>=pow(10, i-1)?y-pow(10, i-1)+1:0));
        }
        return cnt;
    }
};
```

### [119. 杨辉三角 II](https://leetcode-cn.com/problems/pascals-triangle-ii/)

难度：简单 # 2020.02.12

给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。

![PascalTriangleAnimated2](./images/PascalTriangleAnimated2.gif)

在杨辉三角中，每个数是它左上方和右上方的数的和。

**示例：**
```
输入: 3
输出: [1,3,3,1]
```
**进阶：**

+ 你可以优化你的算法到 O(k) 空间复杂度吗？

```cpp
class Solution {
public:
    vector<int> getRow(int rowIndex) { // 二项式几何解释
        vector<int> row(rowIndex + 1);
        row[0] = 1;
        for (int i = 1; i <= rowIndex; ++i) {
            row[i] = 1LL * row[i - 1] * (rowIndex - i + 1) / i;
        }
        return row;
    }
};
```



## 树

### [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

难度：简单 # 2020.09.28

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：
```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```
镜像输出：
```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

**示例 1：**

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]


**限制：**

`0 <= 节点个数 <= 1000`

注意：本题与主站 226 题相同：https://leetcode-cn.com/problems/invert-binary-tree/

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(root == nullptr) return nullptr;
        TreeNode *tmp = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(tmp);
        return root;
    }
};
```

### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

难度：简单 # 2020.09.29

将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

**示例：**

给定有序数组: [-10,-3,0,5,9],

一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
```
      0
     / \
   -3   9
   /   /
 -10  5
```
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return organize(nums, 0, nums.size()-1);
    }
    // 二叉搜索树，递归将每段中间靠左的值作为树根
    TreeNode* organize(vector<int>& nums, int l, int r){
        if(l <= r){
            int mid = (l + r) / 2;
            TreeNode *root = new TreeNode(nums[mid]);
            root->left = organize(nums, l, mid-1);
            root->right = organize(nums, mid+1, r);
            return root;
        }
        return nullptr;
    }
};
```

### [979. 在二叉树中分配硬币](https://leetcode-cn.com/problems/distribute-coins-in-binary-tree/)

难度：中等 # 远古

给定一个有 `N` 个结点的二叉树的根结点 `root`，树中的每个结点上都对应有 `node.val` 枚硬币，并且总共有 `N` 枚硬币。

在一次移动中，我们可以选择两个相邻的结点，然后将一枚硬币从其中一个结点移动到另一个结点。(移动可以是从父结点到子结点，或者从子结点移动到父结点。)。

返回使每个结点上只有一枚硬币所需的移动次数。

**示例 1：**
<img src="./images/tree1.png" alt="tree1" style="zoom:50%;" />

```
输入：[3,0,0]
输出：2
解释：从树的根结点开始，我们将一枚硬币移到它的左子结点上，一枚硬币移到它的右子结点上。
```
**示例 2：**
<img src="./images/tree2.png" alt="tree2" style="zoom:50%;" />

```
输入：[0,3,0]
输出：3
解释：从根结点的左子结点开始，我们将两枚硬币移到根结点上 [移动两次]。然后，我们把一枚硬币从根结点移到右子结点上。
```
**示例 3：**
<img src="./images/tree3.png" alt="tree3" style="zoom:50%;" />

```
输入：[1,0,2]
输出：2
```
**示例 4：**
<img src="./images/tree4.png" alt="tree4" style="zoom:50%;" />

```
输入：[1,0,0,null,3]
输出：4
```

**提示：**

1. `1<= N <= 100`
2. `0 <= node.val <= N`

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 class Solution {
 public:
    int distributeCoins(TreeNode* root) {
        int ans = 0;
        dfs(root, ans);
        return ans;
    }
    // 返回当前节点向上传递的值
    int dfs(TreeNode* root, int &ans){
        if(root == nullptr) return 0;
        int L = dfs(root->left, ans);
        int R = dfs(root->right, ans);
        ans += abs(L) + abs(R);
        return root->val + L + R - 1;
    }
 };
```

### [1457. 二叉树中的伪回文路径](https://leetcode-cn.com/problems/pseudo-palindromic-paths-in-a-binary-tree/)

难度：中等 # 2020.10.01

给你一棵二叉树，每个节点的值为 1 到 9 。我们称二叉树中的一条路径是 「**伪回文**」的，当它满足：路径经过的所有节点值的排列中，存在一个回文序列。

请你返回从根到叶子节点的所有路径中 **伪回文** 路径的数目。

**示例 1：**
![palindromic_paths_1](./images/palindromic_paths_1.png)

```
输入：root = [2,3,1,3,1,null,1]
输出：2 
解释：上图为给定的二叉树。总共有 3 条从根到叶子的路径：红色路径 [2,3,3] ，绿色路径 [2,1,1] 和路径 [2,3,1] 。
     在这些路径中，只有红色和绿色的路径是伪回文路径，因为红色路径 [2,3,3] 存在回文排列 [3,2,3] ，绿色路径 [2,1,1] 存在回文排列 [1,2,1] 。
```
**示例 2：**
![palindromic_paths_2](./images/palindromic_paths_2.png)


```
输入：root = [2,1,1,1,3,null,null,null,null,null,1]
输出：1 
解释：上图为给定二叉树。总共有 3 条从根到叶子的路径：绿色路径 [2,1,1] ，路径 [2,1,3,1] 和路径 [2,1] 。
     这些路径中只有绿色路径是伪回文路径，因为 [2,1,1] 存在回文排列 [1,2,1] 。
```
**示例 3：**
```
输入：root = [9]
输出：1
```
**提示：**

+ 给定二叉树的节点数目在 `1` 到 `10^5` 之间。
+ 节点值在 `1` 到 `9` 之间。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
 class Solution {
 public:
    bool judge(unordered_map<int, int> &his){
        int cnt = 0;
        for(auto iter = his.begin(); iter != his.end(); iter++){
            if(iter->second % 2 == 1) cnt++;
        }
        if(cnt > 1) return false;
        return true;
    }
    void dfs(TreeNode* root, unordered_map<int, int> &his, int &ans){
        if(root == nullptr) return;
        his[root->val]++;
        if(root->left == nullptr && root->right == nullptr) if(judge(his)) ans++;
        dfs(root->left, his, ans);
        dfs(root->right, his, ans);
        his[root->val]--;
        return;
    }
    int pseudoPalindromicPaths (TreeNode* root) {
        unordered_map<int, int> his;
        int ans = 0;
        dfs(root, his, ans);
        return ans;
    }
 };
```

### [1130. 叶值的最小代价生成树](https://leetcode-cn.com/problems/minimum-cost-tree-from-leaf-values/)

难度：中等 # 2020.10.03

给你一个正整数数组 `arr`，考虑所有满足以下条件的二叉树：

每个节点都有 0 个或是 2 个子节点。
数组 `arr` 中的值与树的中序遍历中每个叶节点的值一一对应。（知识回顾：如果一个节点有 0 个子节点，那么该节点为叶节点。）
每个非叶节点的值等于其左子树和右子树中叶节点的最大值的乘积。
在所有这样的二叉树中，返回每个非叶节点的值的最小可能总和。这个和的值是一个 32 位整数。

**示例：**
```
输入：arr = [6,2,4]
输出：32
解释：
有两种可能的树，第一种的非叶节点的总和为 36，第二种非叶节点的总和为 32。

    24            24
   /  \          /  \
  12   4        6    8
 /  \               / \
6    2             2   4
```
**提示：**

+ `2 <= arr.length <= 40`
+ `1 <= arr[i] <= 15`
+ 答案保证是一个 32 位带符号整数，即小于 `2^31`。

```cpp
class Solution {
public:
    int mctFromLeafValues(vector<int>& arr) {
        int len = arr.size();
        // 存每一段最大值
        vector<vector<int>> max_val(len, vector<int>(len, 0));
        for(int i=0; i<len; i++){
            for(int j=i; j<len; j++){
                for(int k=i; k<=j; k++){
                    if(arr[k] > max_val[i][j]) max_val[i][j] = arr[k];
                }
            }
        }
        // 长度从小到大动规
        vector<vector<int>> dp(len, vector<int>(len, INT_MAX));
        for(int i=0; i<len; i++){// 很机智，满足转移方程
            dp[i][i] = 0;
        }
        for(int l=1; l<len; l++){// 长度
            for(int i=0; i<len-l; i++){// 起始点
                for(int k=i; k<i+l; k++){// 分隔点
                    dp[i][i+l] = min(dp[i][i+l], dp[i][k] + dp[k+1][i+l] + max_val[i][k] * max_val[k+1][i+l]);
                }
            }
        }
        return dp[0][len-1];
    }
};
```

### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

难度：困难 # 2020.10.03

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 `root`，返回其 **最大路径和** 。

**示例 1：**
![exx1](./images/exx1.jpg)

```
输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
```
**示例 2：**
![exx2](./images/exx2.jpg)

```
输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
```
**提示：**

+ 树中节点数目范围是 `[1, 3 * 104]`
+ `-1000 <= Node.val <= 1000`

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 class Solution {
 public:
    int dfs(TreeNode* root, int &max_val){// 返回到该顶点最大上升路径和
        if(root == nullptr) return 0;
        int l = dfs(root->left, max_val);
        int r = dfs(root->right, max_val);
        int path = max(root->val, root->val + max(l, r));// 该节点、该节点+左子树、该节点+右子树
        max_val = max(max_val, max(path, root->val + l + r));// 若最大路径和经过该节点，该节点+左右子树
        return path;
    }
    int maxPathSum(TreeNode* root) {
        int max_val = INT_MIN;
        dfs(root, max_val);
        return max_val;
    }
 };
```

### [968. 监控二叉树](https://leetcode-cn.com/problems/binary-tree-cameras/)

难度：困难 # 远古

给定一个二叉树，我们在树的节点上安装摄像头。

节点上的每个摄影头都可以监视**其父对象、自身及其直接子对象**。

计算监控树的所有节点所需的最小摄像头数量。

**示例 1：**
![bst_cameras_01](./images/bst_cameras_01.png)

```
输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。
```
**示例 2：**
![bst_cameras_02](./images/bst_cameras_02.png)

```
输入：[0,0,null,0,null,0,null,null,0]
输出：2
解释：需要至少两个摄像头来监视树的所有节点。 上图显示了摄像头放置的有效位置之一。
```
**提示：**

1. 给定树的节点数的范围是 `[1, 1000]`。
2. 每个节点的值都是 0。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 class Solution {
 public:
    int findnum(TreeNode* root){
        if(!root->left && !root->right){
            // cout<<root->val;
            return 0;
        }
        int l = root->left?findnum(root->left):0;
        int r = root->right?findnum(root->right):0;
        if((root->left!=nullptr) && (root->right!=nullptr)){
            switch(root->left->val * root->right->val){// 0代表没有被监控 1代表子节点中有摄像头，自己正被子节点监控 2代表自己就是摄像头
                case 0:
                    root->val = 2;
                    break;
                case 2:
                    root->val = 1;
                    break;
                case 4:
                    root->val = 1;
                    break;
                default:
                    root->val = 0;
            }
        }
        else if(root->left==nullptr){
            switch(root->right->val){
                case 0:
                    root->val = 2;
                    break;
                case 2:
                    root->val = 1;
                    break;
                default:
                    root->val = 0;
            }
        }
        else if(root->right==nullptr){
            switch(root->left->val){
                case 0:
                    root->val = 2;
                    break;
                case 2:
                    root->val = 1;
                    break;
                default:
                    root->val = 0;
            }
        }
        // cout<<root->val;
        return l + r + (root->val==2?1:0);
    }
    int minCameraCover(TreeNode* root) {
        if(!root->left && !root->right)
            return 1;
        int ans = findnum(root) + (root->val==0?1:0);
        return ans;
    }
 };
```

### [173. 二叉搜索树迭代器](https://leetcode-cn.com/problems/binary-search-tree-iterator/)

难度：中等 # 2021.03.28

实现一个二叉搜索树迭代器类 `BSTIterator` ，表示一个按中序遍历二叉搜索树（BST）的迭代器：

+ `BSTIterator(TreeNode root)` 初始化 `BSTIterator` 类的一个对象。BST 的根节点 `root` 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 `BST 中的任何元素。
+ `boolean hasNext()` 如果向指针右侧遍历存在数字，则返回 `true` ；否则返回 `false` 。
+ `int next()` 将指针向右移动，然后返回指针处的数字。
注意，指针初始化为一个不存在于 BST 中的数字，所以对 `next()` 的首次调用将返回 BST 中的最小元素。

你可以假设 `next()` 调用总是有效的，也就是说，当调用 `next()` 时，BST 的中序遍历中至少存在一个下一个数字。

**示例：**
![bst-tree](./images/bst-tree.png)

```
输入
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
输出
[null, 3, 7, true, 9, true, 15, true, 20, false]

解释
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // 返回 3
bSTIterator.next();    // 返回 7
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 9
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 15
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 20
bSTIterator.hasNext(); // 返回 False
```

**提示：**

+ 树中节点的数目在范围 `[1, 10^5]` 内
+ `0 <= Node.val <= 10^6`
+ 最多调用 `10^5` 次 `hasNext` 和 `next` 操作

**进阶：**

+ 你可以设计一个满足下述条件的解决方案吗？`next()` 和 `hasNext()` 操作均摊时间复杂度为 `O(1)` ，并使用 `O(h)` 内存。其中 `h` 是树的高度。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
 class BSTIterator {
 public:
    vector<int> arr;
    int idx;
    void inOrder(TreeNode* root) {
        if(!root) return;
        inOrder(root->left);
        arr.push_back(root->val);
        inOrder(root->right);
    }

    BSTIterator(TreeNode* root) {
        idx = 0;
        inOrder(root);
    }
    
    int next() {
        return arr[idx++];
    }
    
    bool hasNext() {
        return (idx < arr.size());
    }
 };

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator* obj = new BSTIterator(root);
 * int param_1 = obj->next();
 * bool param_2 = obj->hasNext();
 */
```

### [530. 二叉搜索树的最小绝对差](https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/)

难度：简单 # 2021.04.13

给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。

**示例：**
```
输入：

   1
    \
     3
    /
   2

输出：
1

解释：
最小绝对差为 1，其中 2 和 1 的差的绝对值为 1（或者 2 和 3）。
```

**提示：**

+ 树中至少有 2 个节点。
+ 本题与 783 https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/ 相同

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
 class Solution {
 public:
    void dfs(TreeNode* root, vector<int> &nums){
        if(root){
            dfs(root->left, nums);
            nums.push_back(root->val);
            dfs(root->right, nums);
        }
    }
    int getMinimumDifference(TreeNode* root) {
        vector<int> nums;
        dfs(root, nums);
        int ans = INT_MAX;
        for(int i=1; i<nums.size(); i++){
            ans = min(ans, nums[i]-nums[i-1]);
        }
        return ans;
    }
 };
```

### [208. 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

难度：中等 # 2021.04.14

Trie（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

+ `Trie()` 初始化前缀树对象。
+ `void insert(String word)` 向前缀树中插入字符串 `word` 。
+ `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
+ `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

**示例：**
```
输入
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出
[null, null, true, false, true, null, true]

解释
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

**提示：**

+ `1 <= word.length, prefix.length <= 2000`
+ `word` 和 `prefix` 仅由小写英文字母组成
+ `insert`、`search` 和 `startsWith` 调用次数 **总计** 不超过 `3 * 10^4` 次

```cpp
class Trie {
private:
    vector<Trie*> children;
    bool isEnd;

    Trie* searchPrefix(string prefix) {
        Trie* node = this;
        for (char ch : prefix) {
            ch -= 'a';
            if (node->children[ch] == nullptr) {
                return nullptr;
            }
            node = node->children[ch];
        }
        return node;
    }

public:
    /** Initialize your data structure here. */
    Trie() : children(26), isEnd(false) {}
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        Trie* node = this;
        for (char ch : word) {
            ch -= 'a';
            if (node->children[ch] == nullptr) {
                node->children[ch] = new Trie();
            }
            node = node->children[ch];
        }
        node->isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        Trie* node = this->searchPrefix(word);
        return node != nullptr && node->isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        return this->searchPrefix(prefix) != nullptr;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```

### [897. 递增顺序搜索树](https://leetcode-cn.com/problems/increasing-order-search-tree/)

难度：简单 # 2021.04.25

给你一棵二叉搜索树，请你 按中序遍历 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

**示例 1：**

![897ex1](./images/897ex1.jpg)
```
输入：root = [5,3,6,2,4,null,8,1,null,null,null,7,9]
输出：[1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]
```
**示例 2：**

![897ex2](./images/897ex2.jpg)
```
输入：root = [5,1,7]
输出：[1,null,5,null,7]
```

**提示：**

+ 树中节点数的取值范围是 `[1, 100]`
+ `0 <= Node.val <= 1000`

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
 class Solution {
 private:
    TreeNode *last;
 public:
    void inorder(TreeNode *node){
        if(node == nullptr){
            return;
        }
        inorder(node->left);
        // 在中序遍历的过程中修改节点指向
        last->right = node;
        node->left = nullptr;
        last = node;
        inorder(node->right);
    }
    TreeNode *increasingBST(TreeNode *root) {
        TreeNode *head = new TreeNode(-1);
        last = head;
        inorder(root);
        return head->right;
    }
 };
```

### [938. 二叉搜索树的范围和](https://leetcode-cn.com/problems/range-sum-of-bst/)

难度：简单 # 2021.04.27

给定二叉搜索树的根结点 `root`，返回值位于范围 `[low, high]` 之间的所有结点的值的和。

**示例 1：**

![bst1](./images/bst1.jpg)
```
输入：root = [10,5,15,3,7,null,18], low = 7, high = 15
输出：32
```
**示例 2：**

![bst2](./images/bst2.jpg)
```
输入：root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
输出：23
```

**提示：**

+ 树中节点数目在范围 `[1, 2 * 10^4]` 内
+ `1 <= Node.val <= 10^5`
+ `1 <= low <= high <= 10^5`
+ 所有 `Node.val` **互不相同**

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
 class Solution {
 public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        if(!root) return 0;
        if(root->val < low) return rangeSumBST(root->right, low, high);
        else if(root->val > high) return rangeSumBST(root->left, low, high);
        else return root->val + rangeSumBST(root->left, low, high) + rangeSumBST(root->right, low, high);
    }
 };
```



## 贪心

### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

难度：简单 # 2020.10.12

给定一个数组，它的第 *i* 个元素是一支给定股票第 *i* 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**示例 1：**
```
输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
```
**示例 2：**
```
输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
```
**示例 3：**
```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```
**提示：**

+ `1 <= prices.length <= 3 * 10 ^ 4`
+ `0 <= prices[i] <= 10 ^ 4`

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        int ans = 0;
        for(int i=0; i<len-1; i++){
            if(prices[i] < prices[i+1]) ans += prices[i+1] - prices[i];
        }
        return ans;
    }
};
```

### [134. 加油站](https://leetcode-cn.com/problems/gas-station/)

难度：中等 # 2020.10.13

在一条环路上有 *N* 个加油站，其中第 *i* 个加油站有汽油 `gas[i]` 升。

你有一辆油箱容量无限的的汽车，从第 *i* 个加油站开往第 *i+1* 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

**说明：**

+ 如果题目有解，该答案即为唯一答案。
+ 输入数组均为非空数组，且长度相同。
+ 输入数组中的元素均为非负数。

**示例 1：**

```
输入: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

输出: 3

解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。
```
**示例 2：**
```
输入: 
gas  = [2,3,4]
cost = [3,4,3]

输出: -1

解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。
```

```cpp
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int len = gas.size();
        int remain = 0;
        int sum = 0;
        int start = 0;
        for(int i=0; i<len; i++){
            // sum 是总和，不能小于0
            sum += gas[i] - cost[i];
            remain += gas[i] - cost[i];
            if(remain < 0){
                start = i + 1;// 小于0说明最后一个站太难过去，从头开始到这个站间的站都应该跳过
                remain = 0;// 从跳过后的站开始重新计算
            }
        }
        return sum >= 0 ? start : -1;
    }
};
```

### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

难度：中等 # 远古

给定一个非负整数数组 `nums`，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

**示例 1：**
```
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```
**示例 2：**
```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```
**提示：**

+ `1 <= nums.length <= 3 * 10^4`
+ `0 <= nums[i] <= 10^5`

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        // 维护最远可达位置
        int k = 0;
        int len = nums.size();
        for(int i=0; i<len; i++){
            if(i > k) return false;
            if(i + nums[i] > k) k = i + nums[i];
        }
        return true;
    }
};
```

### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

难度：中等 # 2020.10.16

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

**注意：**

可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。
**示例 1：**
```
输入: [ [1,2], [2,3], [3,4], [1,3] ]
输出: 1
解释: 移除 [1,3] 后，剩下的区间没有重叠。
```
**示例 2：**
```
输入: [ [1,2], [1,2], [1,2] ]
输出: 2
解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
```
**示例 3：**
```
输入: [ [1,2], [2,3] ]
输出: 0
解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```
```cpp
class Solution {
public:
    static bool cmp(vector<int> a, vector<int> b){// 按左区间排序
        return a[0] < b[0];
    }
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        int len = intervals.size();
        sort(intervals.begin(), intervals.end(), cmp);
        int i = 0, j = 1, ans = 0;
        while(j < len){// 三种情况
            if(intervals[i][1] <= intervals[j][0]){// 前后不相交
                i=j; j++;
                continue;
            }
            else if(intervals[i][1] >= intervals[j][1]){// 前包含后
                ans++; i=j; j++;
                continue;
            }
            else if(intervals[i][1] < intervals[j][1]){// 前后相交
                ans++; j++;
                continue;
            }
        }
        return ans;
    }
};
```

### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

难度：中等 # 2020.10.16

假设有打乱顺序的一群人站成一个队列，数组 `people` 表示队列中一些人的属性（不一定按顺序）。每个 `people[i] = [hi, ki]` 表示第 `i` 个人的身高为 `hi` ，前面 **正好** 有 `ki` 个身高大于或等于 `hi` 的人。

请你重新构造并返回输入数组 `people` 所表示的队列。返回的队列应该格式化为数组 `queue`，其中 `queue[j] = [hj, kj]` 是队列中第 `j` 个人的属性（`queue[0]` 是排在队列前面的人）。

**示例 1：**
```
输入：people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
输出：[[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
解释：
编号为 0 的人身高为 5 ，没有身高更高或者相同的人排在他前面。
编号为 1 的人身高为 7 ，没有身高更高或者相同的人排在他前面。
编号为 2 的人身高为 5 ，有 2 个身高更高或者相同的人排在他前面，即编号为 0 和 1 的人。
编号为 3 的人身高为 6 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
编号为 4 的人身高为 4 ，有 4 个身高更高或者相同的人排在他前面，即编号为 0、1、2、3 的人。
编号为 5 的人身高为 7 ，有 1 个身高更高或者相同的人排在他前面，即编号为 1 的人。
因此 [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] 是重新构造后的队列。
```
**示例 2：**
```
输入：people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
输出：[[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
```
**提示：**

+ `1 <= people.length <= 2000`
+ `0 <= hi <= 10^6`
+ `0 <= ki < people.length`
+ 题目数据确保队列可以被重建

```cpp
class Solution {
public:
    static bool cmp(vector<int> a, vector<int> b){// 7 0, 7 1, 6 1, 5 0, 5 2, 4 4
        if(a[0] == b[0]) return a[1] < b[1];
        return a[0] > b[0];
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        // 个子高的人看不见个子矮的人，从高到低按照索引
        int len = people.size();
        sort(people.begin(), people.end(), cmp);
        for(int i=0; i<len; i++){// i索引
            if(people[i][1] != i){
                vector<int> temp = people[i];
                int start = people[i][1];
                int j = i - 1;
                while(j >= start){
                    people[j+1] = people[j];
                    j--;
                }
                people[start] = temp;
            }
        }
        return people;
    }
};
```

### [649. Dota2 参议院](https://leetcode-cn.com/problems/dota2-senate/)

难度：中等 # 2020.10.16

Dota2 的世界里有两个阵营：`Radiant`(天辉)和 `Dire`(夜魇)

Dota2 参议院由来自两派的参议员组成。现在参议院希望对一个 Dota2 游戏里的改变作出决定。他们以一个基于轮为过程的投票进行。在每一轮中，每一位参议员都可以行使两项权利中的一项：

1. `禁止一名参议员的权利`：参议员可以让另一位参议员在这一轮和随后的几轮中丧失**所有的权利**。

2. `宣布胜利`：如果参议员发现有权利投票的参议员都是**同一个阵营的**，他可以宣布胜利并决定在游戏中的有关变化。

给定一个字符串代表每个参议员的阵营。字母 “R” 和 “D” 分别代表了 `Radiant`（天辉）和 `Dire`（夜魇）。然后，如果有 `n` 个参议员，给定字符串的大小将是 `n`。

以轮为基础的过程从给定顺序的第一个参议员开始到最后一个参议员结束。这一过程将持续到投票结束。所有失去权利的参议员将在过程中被跳过。

假设每一位参议员都足够聪明，会为自己的政党做出最好的策略，你需要预测哪一方最终会宣布胜利并在 Dota2 游戏中决定改变。输出应该是 `Radiant` 或 `Dire`。

**示例 1：**
```
输入："RD"
输出："Radiant"
解释：第一个参议员来自 Radiant 阵营并且他可以使用第一项权利让第二个参议员失去权力，因此第二个参议员将被跳过因为他没有任何权利。然后在第二轮的时候，第一个参议员可以宣布胜利，因为他是唯一一个有投票权的人
```
**示例 2：**
```
输入："RDD"
输出："Dire"
解释：
第一轮中,第一个来自 Radiant 阵营的参议员可以使用第一项权利禁止第二个参议员的权利
第二个来自 Dire 阵营的参议员会被跳过因为他的权利被禁止
第三个来自 Dire 阵营的参议员可以使用他的第一项权利禁止第一个参议员的权利
因此在第二轮只剩下第三个参议员拥有投票的权利,于是他可以宣布胜利
```

**提示：**

+ 给定字符串的长度在 `[1, 10,000]` 之间.

```cpp
class Solution {
public:
    string predictPartyVictory(string senate) {
        int len = senate.size();
        bool R = true, D = true;
        int slash = 0;// 正时天辉ban夜魇，负时夜魇ban天辉
        while(R && D){
            R = false; D = false;
            for(int i=0; i<len; i++){
                if(senate[i] == 'R'){
                    R = true;
                    if(slash < 0){
                        senate[i] = 'X';
                        slash++;
                    }else slash++;
                }else if(senate[i] == 'D'){
                    D = true;
                    if(slash > 0){
                        senate[i] = 'X';
                        slash--;
                    }else slash--;
                }
            }
        }
        if(R) return "Radiant";
        else return "Dire";
    }
};
```

### [402. 移掉K位数字](https://leetcode-cn.com/problems/remove-k-digits/)

难度：中等 # 2020.10.17

给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。

**注意：**

+ num 的长度小于 10002 且 ≥ k。
+ num 不会包含任何前导零。

**示例 1：**
```
输入: num = "1432219", k = 3
输出: "1219"
解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。
```
**示例 2：**
```
输入: num = "10200", k = 1
输出: "200"
解释: 移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。
```
**示例 3：**
```
输入: num = "10", k = 2
输出: "0"
解释: 从原数字移除所有的数字，剩余为空就是0。
```

```cpp
class Solution {
public:
    string removeKdigits(string num, int k) {// 对于第一位，0-k中最小的一个，第二位就是第一位的后一个索引到k+1中最小的一个，依次类推
        int len = num.size();
        if(k >= len) return "0";
        vector<int> index(len-k, 0);// 每一位的索引
        int from = 0, to = k;// 每一位在这个选择范围中选最小的
        for(int i=0; i<len-k; i++){
            // cout<<"from: "<<from<<" to: "<<to<<endl;
            int minm = INT_MAX;
            for(int j=from; j<=to; j++){
                if(num[j] - '0' < minm){
                    minm = num[j] - '0';
                    index[i] = j;
                    from = j + 1;
                }
            }
            to++;
        }
        string ans;
        for(int i=0; i<len-k; i++){
            if(ans.size() == 0 && num[index[i]] == '0') continue;
            ans += num[index[i]];
        }
        if(ans.size() == 0) return "0";
        return ans;
    }
};
```

### [605. 种花问题](https://leetcode-cn.com/problems/can-place-flowers/)

难度：简单 # 2021.01.01

假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。

给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 **n** 。能否在不打破种植规则的情况下种入 **n** 朵花？能则返回True，不能则返回False。

**示例 1：**
```
输入: flowerbed = [1,0,0,0,1], n = 1
输出: True
```
**示例 2：**
```
输入: flowerbed = [1,0,0,0,1], n = 2
输出: False
```
**注意：**

1. 数组内已种好的花不会违反种植规则。
2. 输入的数组长度范围为 [1, 20000]。
3. **n** 是非负整数，且不会超过输入数组的大小。

```cpp
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        int cnt = 0;
        int len = flowerbed.size();
        int prev = -1;
        for (int i=0; i<len; i++){
            if (flowerbed[i] == 1) {
                if (prev < 0) {
                    cnt += i / 2;
                } else {
                    cnt += (i - prev - 2) / 2;
                }
                prev = i;
            }
        }
        if (prev < 0){ // 全0
            cnt += (len + 1) / 2;
        } else { // 末尾一串0
            cnt += (len - prev - 1) / 2;
        }
        return cnt >= n;
    }
};
```



## 双指针

### [125. 验证回文串](https://leetcode-cn.com/problems/valid-palindrome/)

难度：简单 # 2020.10.19

给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

**示例 1：**
```
输入: "A man, a plan, a canal: Panama"
输出: true
```
**示例 2：**
```
输入: "race a car"
输出: false
```

```cpp
class Solution {
public:
    bool isPalindrome(string s) {
        int len = s.size();
        string str;
        for(int i=0; i<len; i++){
            if(isalpha(s[i])) str += tolower(s[i]);
            else if(isdigit(s[i])) str += s[i];
        }
        len = str.size();
        int from = 0, to = len-1;
        while(from < to){
            if(str[from] != str[to]) return false;
            from++; to--;
        }
        return true;
    }
};
```

### [167. 两数之和 II - 输入有序数组](https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/)

难度：简单 # 远古

给定一个已按照 *升序排列* 的有序数组，找到两个数使得它们相加之和等于目标数。

函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。

说明:

+ 返回的下标值（index1 和 index2）不是从零开始的。
+ 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

**示例：**
```
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。
```

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int n = numbers.size();
        vector<int> ans;
        int l=0, h=n-1;
        while(l < h){
            int sum = numbers[l] + numbers[h];
            if(sum == target){
                ans.push_back(l+1);
                ans.push_back(h+1);
                break;
            }else if(sum > target){
                h--;
            }else{
                l++;
            }
        }
        return ans;
    }
};
```

### [424. 替换后的最长重复字符](https://leetcode-cn.com/problems/longest-repeating-character-replacement/)

难度：中等 # 2020.10.20

给你一个仅由大写英文字母组成的字符串，你可以将任意位置上的字符替换成另外的字符，总共可最多替换 *k* 次。在执行上述操作后，找到包含重复字母的最长子串的长度。

注意：字符串长度 和 *k* 不会超过 10^4^。

**示例 1：**
```
输入：s = "ABAB", k = 2
输出：4
解释：用两个'A'替换为两个'B',反之亦然。
```
**示例 2：**
```
输入：s = "AABABBA", k = 1
输出：4
解释：
将中间的一个'A'替换为'B',字符串变为 "AABBBBA"。
子串 "BBBB" 有最长重复字母, 答案为 4。
```

```cpp
class Solution {
public:
    int characterReplacement(string s, int k) {
        int len = s.size();
        int cnt[26];// 当前窗口内所有字母的数量
        int maxcnt = 0;// 历史记录中一个窗口中某字母最大数量
        memset(cnt, 0, sizeof(cnt));
        int left = 0, right = 0;
        for(; right<len; right++){
            cnt[s[right]-'A']++;
            maxcnt = max(maxcnt, cnt[s[right]-'A']);
            if(right - left + 1 > maxcnt + k){// 当窗口无法再向右扩时，左界右移
                cnt[s[left]-'A']--;
                left++;
            }
        }
        return len - left;
    }
};
```

### [524. 通过删除字母匹配到字典里最长单词](https://leetcode-cn.com/problems/longest-word-in-dictionary-through-deleting/)

难度：中等 # 2020.10.21

给定一个字符串和一个字符串字典，找到字典里面最长的字符串，该字符串可以通过删除给定字符串的某些字符来得到。如果答案不止一个，返回长度最长且字典顺序最小的字符串。如果答案不存在，则返回空字符串。

**示例 1：**
```
输入:
s = "abpcplea", d = ["ale","apple","monkey","plea"]
输出: 
"apple"
```
**示例 2：**
```
输入:
s = "abpcplea", d = ["a","b","c"]
输出: 
"a"
```
**说明：**

1. 所有输入的字符串只包含小写字母。
2. 字典的大小不会超过 1000。
3. 所有输入的字符串长度不会超过 1000。

```cpp
class Solution {
public:
    static bool cmp(string a, string b){// 对字符串按照长度递减，字典序递增顺序排序
        int la = a.size(), lb = b.size();
        if(la == lb) return a < b;
        return la > lb;
    }
    bool judge(string s, int slen, string di, int dilen){
        int i=0, j=0;// i为s中索引，j为di中索引
        while(j < dilen){
            if(i >= slen) return false;
            while(i < slen){
                if(di[j] == s[i]){
                    j++; i++;
                    break;
                }
                i++;
            }
        }
        return true;
    }
    string findLongestWord(string s, vector<string>& d) {
        int dlen = d.size();
        int slen = s.size();
        sort(d.begin(), d.end(), cmp);
        for(int i=0; i<dlen; i++){
            if(d[i].size() <= slen){
                if(judge(s, slen, d[i], d[i].size())) return d[i];
            }
        }
        return "";
    }
};
```

### [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

难度：中等 # 远古

给定一个包括 n 个整数的数组 `nums` 和 一个目标值 `target`。找出 `nums` 中的三个整数，使得它们的和与 `target` 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

**示例：**
```
输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
```

**提示：**

+ `3 <= nums.length <= 10^3`
+ `-10^3 <= nums[i] <= 10^3`
+ `-10^4 <= target <= 10^4`

```cpp
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int ans[n];
        memset(ans, 0x7f, sizeof(ans));
        for(int p=0; p<n-2; p++){
            int i = p + 1;
            int j = n - 1;
            while(i < j){
                int sum = nums[p] + nums[i] + nums[j];
                if(sum == target){
                    return target;
                }else if(sum < target){
                    if(target-sum < abs(ans[p]-target)){
                        ans[p] = sum;
                    }
                    i++;
                }else{
                    if(sum-target < abs(ans[p]-target)){
                        ans[p] = sum;
                    }
                    j--;
                }
            }
        }
        int val=abs(ans[0]-target), x=0;
        for(int i=0; i<n-2; i++){
            if(abs(ans[i]-target) < val){
                val = abs(ans[i]-target);
                x = i;
            }
        }
        return ans[x];
    }
};
```

### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

难度：中等 # 远古

给定一个包含 n 个整数的数组 `nums` 和一个目标值 `target`，判断 `nums` 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 `target` 相等？找出所有满足条件且不重复的四元组。

**注意：**

答案中不可以包含重复的四元组。

**示例：**
```
给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
```

```cpp
class Solution {
public:
    void search(vector<int>& nums, int i, vector<int>& his, int& n, set<vector<int> >& s, int target){
        // cout<<"i: "<<i<<endl;
        // for(int j=0; j<his.size(); j++){
        //     cout<<his[j]<<" ";
        // }
        // cout<<endl;
        if(n-i+his.size() < 4)// 加上剩余的不足4个
            return;
        if(his.size() > 3)// 已满4个
            return;
        if(his.size() == 3){// 还剩1个，找到满足的后加入集合
            for(int j=i; j<n; j++){
                if(nums[j] == target){
                    his.push_back(nums[j]);
                    s.insert(his);
                    his.pop_back();
                    return;
                }
            }
        }
        int lsum = 0;
        for(int j=0; j<4-his.size(); j++){// 左界最小的几个和
            lsum += nums[i+j];
        }
        if(target < lsum)// 左界最小的几个和已经大于目标值
            return;
        int hsum = 0;
        for(int j=0; j<4-his.size(); j++){// 右界最大的几个和
            hsum += nums[n-1-j];
        }
        if(target > hsum)// 右界最大的几个和已经小于目标值
            return;
        search(nums, i+1, his, n, s, target);// 不选第i个的情况
        his.push_back(nums[i]);// 选第i个的情况
        search(nums, i+1, his, n, s, target-nums[i]);
        his.pop_back();
    }
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        set<vector<int> > s;
        int n = nums.size();
        vector<int> his;
        search(nums, 0, his, n, s, target);
        vector<vector<int> > ans;
        for(set<vector<int> >::iterator it=s.begin(); it!=s.end(); it++){
            ans.push_back(*it);
        }
        return ans;
    }
};
```

### [986. 区间列表的交集](https://leetcode-cn.com/problems/interval-list-intersections/)

难度：中等 # 2020.10.23

给定两个由一些 **闭区间** 组成的列表，`firstList` 和 `secondList` ，其中 `firstList[i] = [starti, endi]` 而 `secondList[j] = [startj, endj]` 。每个区间列表都是成对 **不相交** 的，并且 **已经排序** 。

返回这 **两个区间列表的交集** 。

形式上，**闭区间** `[a, b]`（其中 `a <= b`）表示实数 `x` 的集合，而 `a <= x <= b` 。

两个闭区间的 **交集** 是一组实数，要么为空集，要么为闭区间。例如，`[1, 3]` 和 `[2, 4]` 的交集为 `[2, 3]` 。

**示例 1：**

![interval1](./images/interval1.png)

```
输入：firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
输出：[[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```
**示例 2：**
```
输入：firstList = [[1,3],[5,9]], secondList = []
输出：[]
```
**示例 3：**
```
输入：firstList = [], secondList = [[4,8],[10,12]]
输出：[]
```
**示例 4：**
```
输入：firstList = [[1,7]], secondList = [[3,10]]
输出：[[3,7]]
```

**提示：**

+ `0 <= firstList.length, secondList.length <= 1000`
+ `firstList.length + secondList.length >= 1`
+ `0 <= starti < endi <= 10^9`
+ `endi < starti+1`
+ `0 <= startj < endj <= 10^9`
+ `endj < startj+1`

```cpp
class Solution {
public:
    vector<vector<int>> intervalIntersection(vector<vector<int>>& A, vector<vector<int>>& B) {
        int Alen = A.size();
        int Blen = B.size();
        int i=0, j=0; // i是A中索引，j是B中索引
        vector<vector<int>> ans; vector<int> tmp(2, 0);
        while(i < Alen && j < Blen){// 讨论B中下一个区间相对于A中当前区间的位置
            if(B[j][1] < A[i][0]){
                // cout<<1<<endl;
                j++;
            }
            else if(B[j][1] >= A[i][0] && B[j][1] <= A[i][1]){
                // cout<<2<<endl;
                tmp[0] = max(A[i][0], B[j][0]);
                tmp[1] = B[j][1];
                ans.push_back(tmp);
                j++;
            }
            else if(B[j][0] <= A[i][1]){
                // cout<<3<<endl;
                tmp[0] = max(A[i][0], B[j][0]);
                tmp[1] = A[i][1];
                ans.push_back(tmp);
                i++;
            }
            else{
                // cout<<4<<endl;
                i++;
            }
        }
        return ans;
    }
};
```

### [141. 环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

难度：简单 # 2021.02.02

给定一个链表，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 `true` 。 否则，返回 `false`。

**进阶：**

你能用 *O(1)*（即，常量）内存解决此问题吗？

**示例 1：**
![circularlinkedlist](./images/circularlinkedlist.png)
```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```
**示例 2：**
![circularlinkedlist_test2](./images/circularlinkedlist_test2.png)
```
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```
**示例 3：**
![circularlinkedlist_test3](./images/circularlinkedlist_test3.png)
```
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

**提示：**

+ 链表中节点的数目范围是 `[0, 10^4]`
+ `-10^5 <= Node.val <= 10^5`
+ `pos` 为 `-1` 或者链表中的一个 **有效索引** 。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow=head;
        if(slow == nullptr) return false;
        ListNode *fast=slow->next;
        while(fast != slow){
            if(fast == nullptr) return false;
            fast = fast->next;
            if(fast == nullptr) return false;
            fast = fast->next;
            slow = slow->next;
        }
        return true;
    }
};
```

### [1208. 尽可能使字符串相等](https://leetcode-cn.com/problems/get-equal-substrings-within-budget/)

难度：中等 # 2021.02.05

给你两个长度相同的字符串，`s` 和 `t`。

将 `s` 中的第 `i` 个字符变到 `t` 中的第 `i` 个字符需要 `|s[i] - t[i]|` 的开销（开销可能为 0），也就是两个字符的 ASCII 码值的差的绝对值。

用于变更字符串的最大预算是 `maxCost`。在转化字符串时，总开销应当小于等于该预算，这也意味着字符串的转化可能是不完全的。

如果你可以将 `s` 的子字符串转化为它在 `t` 中对应的子字符串，则返回可以转化的最大长度。

如果 `s` 中没有子字符串可以转化成 `t` 中对应的子字符串，则返回 `0`。
**示例 1：**
```
输入：s = "abcd", t = "bcdf", cost = 3
输出：3
解释：s 中的 "abc" 可以变为 "bcd"。开销为 3，所以最大长度为 3。
```
**示例 2：**
```
输入：s = "abcd", t = "cdef", cost = 3
输出：1
解释：s 中的任一字符要想变成 t 中对应的字符，其开销都是 2。因此，最大长度为 1。
```
**示例 3：**
```
输入：s = "abcd", t = "acde", cost = 0
输出：1
解释：你无法作出任何改动，所以最大长度为 1。
```

**提示：**

+ `1 <= s.length, t.length <= 10^5`
+ `0 <= maxCost <= 10^6`
+ `s` 和 `t` 都只含小写英文字母。

```cpp
class Solution {
public:
    int equalSubstring(string s, string t, int maxCost) {
        int len = s.size();
        int i=0, j=0, sum=0, ans=0;
        while(j < len){
            int x = abs(s[j]-t[j]);
            if(x > maxCost){
                sum = 0;
                i = j + 1;
                j = i;
                continue;
            }
            if(sum + x <= maxCost){
                sum += x;
                j++;
                ans = max(ans, j-i);
            }
            else{
                ans = max(ans, j-i);
                sum -= abs(s[i]-t[i]);
                i++;
            }
        }
        ans = max(ans, j-i);
        return ans;
    }
};
```

### [1423. 可获得的最大点数](https://leetcode-cn.com/problems/maximum-points-you-can-obtain-from-cards/)

难度：中等 # 2020.02.06

几张卡牌 **排成一行**，每张卡牌都有一个对应的点数。点数由整数数组 `cardPoints` 给出。

每次行动，你可以从行的开头或者末尾拿一张卡牌，最终你必须正好拿 `k` 张卡牌。

你的点数就是你拿到手中的所有卡牌的点数之和。

给你一个整数数组 `cardPoints` 和整数 `k`，请你返回可以获得的最大点数。

**示例 1：**
```
输入：cardPoints = [1,2,3,4,5,6,1], k = 3
输出：12
解释：第一次行动，不管拿哪张牌，你的点数总是 1 。但是，先拿最右边的卡牌将会最大化你的可获得点数。最优策略是拿右边的三张牌，最终点数为 1 + 6 + 5 = 12 。
```
**示例 2：**
```
输入：cardPoints = [2,2,2], k = 2
输出：4
解释：无论你拿起哪两张卡牌，可获得的点数总是 4 。
```
**示例 3：**
```
输入：cardPoints = [9,7,7,9,7,7,9], k = 7
输出：55
解释：你必须拿起所有卡牌，可以获得的点数为所有卡牌的点数之和。
```
**示例 4：**
```
输入：cardPoints = [1,1000,1], k = 1
输出：1
解释：你无法拿到中间那张卡牌，所以可以获得的最大点数为 1 。
```
**示例 5：**
```
输入：cardPoints = [1,79,80,1,1,1,200,1], k = 3
输出：202
```

**提示：**

+ `1 <= cardPoints.length <= 10^5`
+ `1 <= cardPoints[i] <= 10^4`
+ `1 <= k <= cardPoints.length`

```cpp
class Solution {
public:
    int maxScore(vector<int>& cardPoints, int k) {
        int len = cardPoints.size();
        k = len - k; // 转化成连续k个值和最小
        int sum = 0, tmp = 0;
        for(auto& x: cardPoints) sum += x;
        for(int i=0; i<k; i++){
            tmp += cardPoints[i];
        }
        int ans = tmp;
        for(int i=k; i<len; i++){
            tmp += cardPoints[i];
            tmp -= cardPoints[i-k];
            ans = min(ans, tmp);
        }
        return sum - ans;
    }
};
```

### [992. K 个不同整数的子数组](https://leetcode-cn.com/problems/subarrays-with-k-different-integers/)

给定一个正整数数组 `A`，如果 `A` 的某个子数组中不同整数的个数恰好为 `K`，则称 `A` 的这个连续、不一定不同的子数组为好子数组。

（例如，`[1,2,3,1,2]` 中有 `3` 个不同的整数：`1`，`2`，以及 `3`。）

返回 `A` 中好子数组的数目。 

**示例 1：**
```
输入：A = [1,2,1,2,3], K = 2
输出：7
解释：恰好由 2 个不同整数组成的子数组：[1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2].
```
示例 2：
```
输入：A = [1,2,1,3,4], K = 3
输出：3
解释：恰好由 3 个不同整数组成的子数组：[1,2,1,3], [2,1,3], [1,3,4].
```
**提示：**

+ `1 <= A.length <= 20000`
+ `1 <= A[i] <= A.length`
+ `1 <= K <= A.length`

```cpp
class Solution {
public:
    int subarraysWithKDistinct(vector<int>& A, int K) {
        int n = A.size();
        unordered_map<int, int> cnt1, cnt2;
        int tot1 = 0, tot2 = 0;
        int left1 = 0, left2 = 0, right = 0; // 两个左指针用来标记到右指针间有K个不同数的左闭右开区间
        int ans = 0;
        while (right < n) {
            if(!cnt1[A[right]]++){
                tot1++;
            }
            if(!cnt2[A[right]]++){
                tot2++;
            }
            while(tot1 > K){
                cnt1[A[left1]]--;
                if(!cnt1[A[left1]]) tot1--;
                left1++;
            }
            while(tot2 > K-1){
                cnt2[A[left2]]--;
                if(!cnt2[A[left2]]) tot2--;
                left2++;
            }
            ans += left2 - left1;
            right++;
        }
        return ans;
    }
};
```



### [567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)

难度：中等 # 2021.02.10

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的排列。
换句话说，第一个字符串的排列之一是第二个字符串的子串。

**示例1：**
```
输入: s1 = "ab" s2 = "eidbaooo"
输出: True
解释: s2 包含 s1 的排列之一 ("ba").
```

**示例2：**
```
输入: s1= "ab" s2 = "eidboaoo"
输出: False
```

**注意：**

1. 输入的字符串只包含小写字母
2. 两个字符串的长度都在 `[1, 10,000]` 之间

```cpp
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int len1=s1.size(), len2=s2.size();
        if(len1 > len2){
            return false;
        }
        vector<int> cnt1(26), cnt2(26);
        for(int i=0; i<len1; i++){
            cnt1[s1[i]-'a']++;
            cnt2[s2[i]-'a']++;
        }
        if(cnt1 == cnt2) return true;
        for(int i=len1; i<len2; i++) {
            cnt2[s2[i]-'a']++;
            cnt2[s2[i-len1]-'a']--;
            if(cnt1 == cnt2) return true;
        }
        return false;
    }
};
```

### [1004. 最大连续1的个数 III](https://leetcode-cn.com/problems/max-consecutive-ones-iii/)

难度：中等 # 2021.02.19

给定一个由若干 `0` 和 `1` 组成的数组 `A`，我们最多可以将 `K` 个值从 0 变成 1 。

返回仅包含 1 的最长（连续）子数组的长度。

**示例 1：**
```
输入：A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
输出：6
解释： 
[1,1,1,0,0,1,1,1,1,1,1]
粗体数字从 0 翻转到 1，最长的子数组长度为 6。
```
**示例 2：**
```
输入：A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
输出：10
解释：
[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
粗体数字从 0 翻转到 1，最长的子数组长度为 10。
```

**提示：**

1. `1 <= A.length <= 20000`
2. `0 <= K <= A.length`
3. `A[i]` 为 `0` 或 `1`

```cpp
class Solution {
public:
    int longestOnes(vector<int>& A, int K) {
        int len = A.size();
        vector<int> Acc(len+1); // 前缀和
        for(int i=1; i<len+1; i++){
            Acc[i] = Acc[i-1] + A[i-1];
        }
        int ans=0, l=0, r=1;
        for(; r<len+1; r++){ // 滑动窗口
            while(Acc[r]-Acc[l]+K < r-l) l++;
            ans = max(ans, r-l);
        }
        return ans;
    }
};
```

### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode-cn.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)

难度：中等 # 2021.02.21

给你一个整数数组 nums ，和一个表示限制的整数 limit，请你返回最长连续子数组的长度，该子数组中的任意两个元素之间的绝对差必须小于或者等于 limit 。

如果不存在满足条件的子数组，则返回 0 。

**示例 1：**
```
输入：nums = [8,2,4,7], limit = 4
输出：2 
解释：所有子数组如下：
[8] 最大绝对差 |8-8| = 0 <= 4.
[8,2] 最大绝对差 |8-2| = 6 > 4. 
[8,2,4] 最大绝对差 |8-2| = 6 > 4.
[8,2,4,7] 最大绝对差 |8-2| = 6 > 4.
[2] 最大绝对差 |2-2| = 0 <= 4.
[2,4] 最大绝对差 |2-4| = 2 <= 4.
[2,4,7] 最大绝对差 |2-7| = 5 > 4.
[4] 最大绝对差 |4-4| = 0 <= 4.
[4,7] 最大绝对差 |4-7| = 3 <= 4.
[7] 最大绝对差 |7-7| = 0 <= 4. 
因此，满足题意的最长子数组的长度为 2 。
```
**示例 2：**
```
输入：nums = [10,1,2,4,7,2], limit = 5
输出：4 
解释：满足题意的最长子数组是 [2,4,7,2]，其最大绝对差 |2-7| = 5 <= 5 。
```
**示例 3：**
```
输入：nums = [4,2,2,2,4,4,2,2], limit = 0
输出：3
```

**提示：**

+ `1 <= nums.length <= 10^5`
+ `1 <= nums[i] <= 10^9`
+ `0 <= limit <= 10^9`

```cpp
class Solution {
public:
    int longestSubarray(vector<int>& nums, int limit) {
        map<int, int> cnt; // map自带按key排序
        int ans = 1, len = nums.size();
        for(int l=0, r=0; r<len; r++){
            cnt[nums[r]]++;
            while(cnt.rbegin()->first - cnt.begin()->first > limit){
                if(--cnt[nums[l]] == 0)
                    cnt.erase(nums[l]);
                l++;
            }
            ans = max(ans, r-l+1);
        }
        return ans;
    }
};
```

### [26. 删除有序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

难度：简单 # 2021.04.18

给你一个有序数组 `nums` ，请你 原地 删除重复出现的元素，使每个元素 **只出现一次** ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 **修改输入数组** 并在使用 O(1) 额外空间的条件下完成。

**说明：**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以**「引用」**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:
```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```
**示例 1：**
```
输入：nums = [1,1,2]
输出：2, nums = [1,2]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
```
**示例 2：**
```
输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
```

**提示：**

+ `0 <= nums.length <= 3 * 10^4`
+ `-10^4 <= nums[i] <= 10^4`
+ `nums` 已按升序排列

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        int slow, fast, cnt = 0;
        for(slow=0; slow<len; slow=fast){
            for(fast=slow+1; fast<len; fast++){
                if(nums[fast] != nums[slow]){
                    nums[cnt] = nums[slow];
                    break;
                }
            }
            if(fast == len) nums[cnt] = nums[len-1];
            cnt++;
        }
        for(int i=0; i<len-cnt; i++){
            nums.pop_back();
        }
        return cnt;
    }
};
```

### [633. 平方数之和](https://leetcode-cn.com/problems/sum-of-square-numbers/)

难度：中等 # 2021.04.28

给定一个非负整数 `c` ，你要判断是否存在两个整数 `a` 和 `b`，使得 `a2 + b2 = c` 。

**示例 1：**
```
输入：c = 5
输出：true
解释：1 * 1 + 2 * 2 = 5
```
**示例 2：**
```
输入：c = 3
输出：false
```
**示例 3：**
```
输入：c = 4
输出：true
```
**示例 4：**
```
输入：c = 2
输出：true
```
**示例 5：**
```
输入：c = 1
输出：true
```

**提示：**

+ `0 <= c <= 2^31 - 1`

```cpp
class Solution {
public:
    bool judgeSquareSum(int c) { // 双指针
        long l = 0;
        long r = sqrt(c);
        while(l <= r){
            long t = l*l + r*r;
            if(t < c) l++;
            else if(t > c) r--;
            else return true;
        }
        return false;
    }
};
```



## 图

### [1042. 不邻接植花](https://leetcode-cn.com/problems/flower-planting-with-no-adjacent/)

难度：中等 # 2020.10.25

有 `n` 个花园，按从 `1` 到 `n` 标记。另有数组 `paths` ，其中 `paths[i] = [xi, yi]` 描述了花园 `xi` 到花园 `yi` 的双向路径。在每个花园中，你打算种下四种花之一。

另外，所有花园 **最多** 有 3 条路径可以进入或离开.

你需要为每个花园选择一种花，使得通过路径相连的任何两个花园中的花的种类互不相同。

以数组形式返回 **任一** 可行的方案作为答案 `answer`，其中 `answer[i]` 为在第 `(i+1)` 个花园中种植的花的种类。花的种类用  1、2、3、4 表示。保证存在答案。

**示例 1：**
```
输入：n = 3, paths = [[1,2],[2,3],[3,1]]
输出：[1,2,3]
解释：
花园 1 和 2 花的种类不同。
花园 2 和 3 花的种类不同。
花园 3 和 1 花的种类不同。
因此，[1,2,3] 是一个满足题意的答案。其他满足题意的答案有 [1,2,4]、[1,4,2] 和 [3,2,1]
```
**示例 2：**
```
输入：n = 4, paths = [[1,2],[3,4]]
输出：[1,2,1,2]
```
**示例 3：**
```
输入：n = 4, paths = [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]
输出：[1,2,3,4]
```

**提示：**

+ 1 <= n <= 104
+ 0 <= paths.length <= 2 * 104
+ paths[i].length == 2
+ 1 <= xi, yi <= n
+ xi != yi
+ 每个花园 **最多** 有 3 条路径可以进入或离开

```cpp
class Solution {
public:
    vector<int> gardenNoAdj(int n, vector<vector<int>>& paths) {
        int len = paths.size();
        vector<vector<int>> adjacent(n);
        for(int i=0; i<len; i++){
            adjacent[paths[i][0]-1].push_back(paths[i][1]);
            adjacent[paths[i][1]-1].push_back(paths[i][0]);
        }
        vector<int> ans(n, 0);
        for(int i=0; i<n; i++){
            vector<int> tmp(4, 0);
            int adlen = adjacent[i].size();
            for(int j=0; j<adlen; j++){
                if(ans[adjacent[i][j]-1] != 0){ // i的第j个邻居已经着色了
                    tmp[ans[adjacent[i][j]-1]-1] = 1; // 就把邻居中颜色置1
                }
            }
            for(int j=0; j<4; j++){
                if(tmp[j] == 0){
                    ans[i] = j+1;
                    // cout<<i<<": "<<ans[i]<<endl;
                    break;
                }
            }
        }
        return ans;
    }
};
```

### [210. 课程表 II](https://leetcode-cn.com/problems/course-schedule-ii/)

难度：中等 # 远古

现在你总共有 *n* 门课需要选，记为 `0` 到 `n-1`。

在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: `[0,1]`

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

**示例 1：**
```
输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
```
**示例 2：**
```
输入: 4, [[1,0],[2,0],[3,1],[3,2]]
输出: [0,1,2,3] or [0,2,1,3]
解释: 总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
     因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
```
**说明：**

1. 输入的先决条件是由**边缘列表**表示的图形，而不是邻接矩阵。
2. 你可以假定输入的先决条件中没有重复的边。
**提示：**

1. 这个问题相当于查找一个循环是否存在于有向图中。如果存在循环，则不存在拓扑排序，因此不可能选取所有课程进行学习。
2. 通过 DFS 进行拓扑排序 - 一个关于Coursera的精彩视频教程（21分钟），介绍拓扑排序的基本概念。
3. 拓扑排序也可以通过 BFS 完成。

```cpp
class Solution {
public:
    bool dfs(int u, vector<int>& vis, int n, vector<vector<int>>& pre, vector<int>& a){
        vis[u] = -1; // 正在计算
        int len = pre[u].size();
        for(int i=0; i<len; i++){
            if(vis[pre[u][i]] < 0){ // 出现环
                return false;
            }
            else if(!vis[pre[u][i]] && !dfs(pre[u][i], vis, n, pre, a)){ // 没访问过但后续出现了环
                return false;
            }
        }
        vis[u] = 1;
        a.push_back(u);
        return true;
    }
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int> > pre(numCourses, vector<int> (0));
        int len = prerequisites.size();
        for(int i=0; i<len; i++){
            pre[prerequisites[i][0]].push_back(prerequisites[i][1]);
        }
        vector<int> vis(numCourses, 0);
        vector<int> ans;
        for(int i=0; i<numCourses; i++){
            if(!vis[i]){
                if(!dfs(i, vis, numCourses, pre, ans)){
                    if(ans.size() != numCourses)
                        ans.resize(0);
                    return ans;
                }
            }
        }
        return ans;
    }
};
```

### [面试题 04.01. 节点间通路](https://leetcode-cn.com/problems/route-between-nodes-lcci/)

难度：中等 # 2020.10.26

节点间通路。给定有向图，设计一个算法，找出两个节点之间是否存在一条路径。

**示例1：**
```
输入：n = 3, graph = [[0, 1], [0, 2], [1, 2], [1, 2]], start = 0, target = 2
输出：true
```
**示例2：**
```
输入：n = 5, graph = [[0, 1], [0, 2], [0, 4], [0, 4], [0, 1], [1, 3], [1, 4], [1, 3], [2, 3], [3, 4]], start = 0, target = 4
输出 true
```
**提示：**

1. 节点数量n在[0, 1e5]范围内。
2. 节点编号大于等于 0 小于 n。
3. 图中可能存在自环和平行边。

```cpp
class Solution {
public:
    bool dfs(int& start, int& target, vector<int>& vis, int& n, vector<vector<int>>& adjacent){
        if(start == target) return true;
        vis[start] = 1;
        int len = adjacent[start].size();
        for(int i=0; i<len; i++){
            if(!vis[adjacent[start][i]] && dfs(adjacent[start][i], target, vis, n, adjacent)){
                return true;
            }
        }
        return false;
    }
    bool findWhetherExistsPath(int n, vector<vector<int>>& graph, int start, int target) {
        vector<vector<int>> adjacent(n, vector<int>(0));
        int len = graph.size();
        for(int i=0; i<len; i++){
            adjacent[graph[i][0]].push_back(graph[i][1]);
        }
        vector<int> vis(n, 0);
        return dfs(start, target, vis, n, adjacent);
    }
};
```

### [684. 冗余连接](https://leetcode-cn.com/problems/redundant-connection/)

难度：中等 # 2020.10.27

在本问题中, 树指的是一个连通且无环的无向图。

输入一个图，该图由一个有着 N 个节点 (节点值不重复1, 2, ..., N) 的树及一条附加的边构成。附加的边的两个顶点包含在 1 到 N 中间，这条附加的边不属于树中已存在的边。

结果图是一个以 `边` 组成的二维数组。每一个 `边` 的元素是一对 `[u, v]` ，满足 `u < v`，表示连接顶点 `u` 和 `v` 的**无向**图的边。

返回一条可以删去的边，使得结果图是一个有着N个节点的树。如果有多个答案，则返回二维数组中最后出现的边。答案边 `[u, v]` 应满足相同的格式 `u < v`。

**示例 1：**

```
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的无向图为:
  1
 / \
2 - 3
```

**示例 2：**

```
输入: [[1,2], [2,3], [3,4], [1,4], [1,5]]
输出: [1,4]
解释: 给定的无向图为:
5 - 1 - 2
    |   |
    4 - 3
```

**注意：**

+ 输入的二维数组大小在 3 到 1000。
+ 二维数组中的整数在 1 到 N 之间，其中 N 是输入数组的大小。

**更新(2017-09-26)：**
我们已经重新检查了问题描述及测试用例，明确图是**无向**图。对于有向图详见*冗余连接II*。

```cpp
class Solution { // 并查集解法
private:
    int parent[1001];
    // int find_root(int x) {
    //     while (parent[x] != x) {
    //         parent[x] = parent[parent[x]]; // 部分路径压缩
    //         x = parent[x];
    //     }
    //     return x;
    // }
    int find(int x) { // 路径压缩版本
        int r = x;
        while(parent[r] != r)
            r = parent[r];
        int i = x, j;
        while(i != r){ // 路径压缩
            j = parent[i]; // 在改变上级之前用临时变量j记录下他的值 
            parent[i] = r; // 把上级改为根节点
            i = j;
        }
        return r;
    }
    bool union_root(int x, int y) {
        // int root_x = find_root(x);
        // int root_y = find_root(y);
        int root_x = find(x);
        int root_y = find(y);
        if (root_x == root_y) {
            return false;
        }
        parent[root_x] = root_y;
        return true;
    }    
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int len = edges.size();
        for(int i=1; i<=len; i++){ // 这样记方便
            parent[i] = i;
        }
        for(auto edge : edges){
            if (!union_root(edge[0], edge[1])){
                return edge;
            }
        }
        return {};
    }
};

class Solution { // dfs
public:
    bool dfs(int& start, int& target, vector<int>& vis, vector<vector<int>>& adjacent){
        vis[start-1] = 1;
        if(start == target) return true;
        int len = adjacent[start-1].size();
        for(int i=0; i<len; i++){
            if(!vis[adjacent[start-1][i]-1] && dfs(adjacent[start-1][i], target, vis, adjacent)) return true;
        }
        return false;
    }
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int len = edges.size();
        vector<vector<int>> adjacent(len, vector<int> (0));
        for(int i=0; i<len; i++){ // 无向图邻接表
            adjacent[edges[i][0]-1].push_back(edges[i][1]);
            adjacent[edges[i][1]-1].push_back(edges[i][0]);
        }
        for(int i=len-1; i>=0; i--){ // 从后开始遍历每一个边，是否没有该边依然能找到一条路径
            int start = edges[i][0];
            int target = edges[i][1];
            vector<int> vis(len, 0);
            vis[start-1] = 1;
            int adnum = adjacent[start-1].size();
            for(int j=0; j<adnum; j++){
                if(adjacent[start-1][j] != target && !vis[adjacent[start-1][j]-1]){ // 排除该边
                    if(dfs(adjacent[start-1][j], target, vis, adjacent)) return edges[i];
                }
            }
        }
        return vector<int>(2, 0);
    }
};
```

### [1557. 可以到达所有点的最少点数目](https://leetcode-cn.com/problems/minimum-number-of-vertices-to-reach-all-nodes/)

难度：中等 # 2020.10.29

给你一个 **有向无环图**， `n` 个节点编号为 `0` 到 `n-1` ，以及一个边数组 `edges` ，其中 `edges[i] = [fromi, toi]` 表示一条从点  `fromi` 到点 `toi` 的有向边。

找到最小的点集使得从这些点出发能到达图中所有点。题目保证解存在且唯一。

你可以以任意顺序返回这些节点编号。

 

**示例 1：**

![5480e1](./images/5480e1.png)
```
输入：n = 6, edges = [[0,1],[0,2],[2,5],[3,4],[4,2]]
输出：[0,3]
解释：从单个节点出发无法到达所有节点。从 0 出发我们可以到达 [0,1,2,5] 。从 3 出发我们可以到达 [3,4,2,5] 。所以我们输出 [0,3] 。
```
**示例 2：**

![5480e2](./images/5480e2.png)
```
输入：n = 5, edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]
输出：[0,2,3]
解释：注意到节点 0，3 和 2 无法从其他节点到达，所以我们必须将它们包含在结果点集中，这些点都能到达节点 1 和 4 。
```

**提示：**

+ 2 <= n <= 10^5
+ 1 <= edges.length <= min(10^5, n * (n - 1) / 2)
+ edges[i].length == 2
+ 0 <= fromi, toi < n
+ 所有点对 `(fromi, toi)` 互不相同。

```cpp
class Solution {
public:
    vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) { // 只算上入度为0的点
        int len = edges.size();
        vector<bool> indegree(n, false);
        for(auto edge : edges){
            indegree[edge[1]] = true;
        }
        vector<int> ans;
        for(int i=0; i<n; i++){
            if(indegree[i] == false) ans.push_back(i);
        }
        return ans;
    }
};
```

### [765. 情侣牵手](https://leetcode-cn.com/problems/couples-holding-hands/)

难度：困难 # 2020.10.30

N 对情侣坐在连续排列的 2N 个座位上，想要牵到对方的手。 计算最少交换座位的次数，以便每对情侣可以并肩坐在一起。 一次交换可选择任意两人，让他们站起来交换座位。

人和座位用 `0` 到 `2N-1` 的整数表示，情侣们按顺序编号，第一对是 `(0, 1)`，第二对是 `(2, 3)`，以此类推，最后一对是 `(2N-2, 2N-1)`。

这些情侣的初始座位  `row[i]` 是由最初始坐在第 i 个座位上的人决定的。

**示例 1：**
```
输入: row = [0, 2, 1, 3]
输出: 1
解释: 我们只需要交换row[1]和row[2]的位置即可。
```
**示例 2：**
```
输入: row = [3, 2, 0, 1]
输出: 0
解释: 无需交换座位，所有的情侣都已经可以手牵手了。
```
**说明：**

1. `len(row)` 是偶数且数值在 `[4, 60]` 范围内。
2. 可以保证 `row` 是序列 `0...len(row)-1` 的一个全排列。

```cpp
class Solution {
public:
    vector<int> parent;
    int find_root(int x)
    {
        int r = x;
        while(r != parent[r]){
            r = parent[r];
        }
        int i=x, j;
        while(i != r){
            j = parent[i];
            parent[i] = r;
            i = j;
        }
        return r;
    }
    void union_root(int x, int y)
    {
        int xroot = find_root(x);
        int yroot = find_root(y);
        if (xroot != yroot){
            parent[xroot] = yroot;
        }
    }
    int minSwapsCouples(vector<int> &row)
    {
        int len2 = row.size();
        int len = len2/2;
        parent = vector<int>(len, 0); // 并查集，row[i]/2 对应一个门派
        for (int i=0; i<len; i++){ // 爹初始化成自己
            parent[i] = i;
        }
        for (int i=0; i<len2; i+=2){ // 遍历每一对位置
            union_root(row[i]/2, row[i+1]/2); // 一对位置内两个点的爹一样，不处理，爹不一样就把他们爹变一样
        }
        int ans = 0;
        for (int i=0; i<len; i++){
            if (i != find_root(i)){ // 爹不是自己，说明在别人的环里，需要跳脱出来
                ans++;
            }
        }
        return ans;
    }
};
```

### [685. 冗余连接 II](https://leetcode-cn.com/problems/redundant-connection-ii/)

难度：困难 # 2020.10.31

在本问题中，有根树指满足以下条件的**有向**图。该树只有一个根节点，所有其他节点都是该根节点的后继。每一个节点只有一个父节点，除了根节点没有父节点。

输入一个有向图，该图由一个有着 N 个节点 (节点值不重复 1, 2, ..., N) 的树及一条附加的边构成。附加的边的两个顶点包含在 1 到 N 中间，这条附加的边不属于树中已存在的边。

结果图是一个以 `边` 组成的二维数组。 每一个 `边` 的元素是一对 `[u, v]`，用以表示有向图中连接顶点 `u` 和顶点 `v` 的边，其中 `u` 是 `v` 的一个父节点。

返回一条能删除的边，使得剩下的图是有 N 个节点的有根树。若有多个答案，返回最后出现在给定二维数组的答案。

**示例 1：**
```
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的有向图如下:
  1
 / \
v   v
2-->3
```
**示例 2：**
```
输入: [[1,2], [2,3], [3,4], [4,1], [1,5]]
输出: [4,1]
解释: 给定的有向图如下:
5 <- 1 -> 2
     ^    |
     |    v
     4 <- 3
```
**注意：**

+ 二维数组大小的在 3 到 1000 范围内。
+ 二维数组中的每个整数在 1 到 N 之间，其中 N 是二维数组的大小。

```cpp
class Solution {
private:
    static const int N = 1010; // 如题：二维数组大小的在3到1000范围内
    int parent[N];
    int n; // 边的数量
    // 并查集初始化
    void init() {
        for (int i = 1; i <= n; ++i) {
            parent[i] = i;
        }
    }
    int find(int u) {
        return u == parent[u] ? u : parent[u] = find(parent[u]);
    }
    bool union_root(int u, int v) {
        u = find(u);
        v = find(v);
        if (u == v) {
            return true;
        }
        parent[v] = u;
        return false;
    }
    // 在有向图里找到删除的那条边，使其变成树
    vector<int> getRemoveEdge(const vector<vector<int>>& edges) {
        init(); // 初始化并查集
        for (int i = 0; i < n; i++) { // 遍历所有的边
            if (union_root(edges[i][0], edges[i][1])) { // 构成有向环了，就是要删除的边
                return edges[i];
            }
        }
        return {};
    }
    // 删一条边之后判断是不是树
    bool isTreeAfterRemoveEdge(const vector<vector<int>>& edges, int deleteEdge) {
        init(); // 初始化并查集
        for (int i = 0; i < n; i++) {
            if (i == deleteEdge) continue;
            if (union_root(edges[i][0], edges[i][1])) { // 构成有向环了，一定不是树
                return false;
            }
        }
        return true;
    }

public:
    vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) { // 要么有入度为2的节点，要么有有向环
        int inDegree[N] = {0}; // 记录节点入度
        n = edges.size(); // 边的数量
        for (int i = 0; i < n; i++) {
            inDegree[edges[i][1]]++; // 统计入度
        }
        vector<int> vec; // 记录入度为2节点的两条边
        // 找入度为2的节点所对应的边，注意要倒叙，因为优先返回最后出现在二维数组中的答案
        for (int i = n - 1; i >= 0; i--) {
            if (inDegree[edges[i][1]] == 2) {
                vec.push_back(i);
            }
        }
        // 如果有入度为2的节点，那么一定是两条边里删一个，看删哪个可以构成树
        if (vec.size() > 0) {
            if (isTreeAfterRemoveEdge(edges, vec[0])) {
                return edges[vec[0]];
            } else {
                return edges[vec[1]];
            }
        }
        // 明确没有入度为2的情况，那么一定有有向环，找到构成环的边返回就可以了
        return getRemoveEdge(edges);
    }
};
```

### [721. 账户合并](https://leetcode-cn.com/problems/accounts-merge/)

难度：中等 # 2021.01.18

给定一个列表 `accounts`，每个元素 `accounts[i]` 是一个字符串列表，其中第一个元素 `accounts[i][0]` 是 名称 *(name)*，其余元素是 *emails* 表示该账户的邮箱地址。

现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。

合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是按顺序排列的邮箱地址。账户本身可以以任意顺序返回。

 **示例 1：**
```
输入：
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
输出：
[["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
解释：
第一个和第三个 John 是同一个人，因为他们有共同的邮箱地址 "johnsmith@mail.com"。 
第二个 John 和 Mary 是不同的人，因为他们的邮箱地址没有被其他帐户使用。
可以以任何顺序返回这些列表，例如答案 [['Mary'，'mary@mail.com']，['John'，'johnnybravo@mail.com']，
['John'，'john00@mail.com'，'john_newyork@mail.com'，'johnsmith@mail.com']] 也是正确的。
```
**提示：**

+ `accounts` 的长度将在 `[1，1000]` 的范围内。
+ `accounts[i]` 的长度将在 `[1，10]` 的范围内。
+ `accounts[i][j]` 的长度将在 `[1，30]` 的范围内。

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    void union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
        }
    }
};

class Solution {
public:
    vector<vector<string>> accountsMerge(vector<vector<string>>& acc) {
        // 作用：存储每个邮箱属于哪个账户 ，同时 在遍历邮箱时，判断邮箱是否出现过
        // 格式：<邮箱，账户id>
        unordered_map<string, int> mail_accountid;
        int len = acc.size();
        Djset ds(len);
        for (int i = 0; i < len; i++) {
            int mlen = acc[i].size();
            for (int j = 1; j < mlen; j++) {
                string mail = acc[i][j];
                if (mail_accountid.find(mail) == mail_accountid.end()) {
                    mail_accountid[mail] = i;
                } else {
                    ds.union_root(i, mail_accountid[mail]); // 曾出现过的话，将对应的账户合并
                }
            }
        }
        // 作用： 存储每个账户下的邮箱
        // 格式： <账户id，邮箱列表> >
        // 注意：这里的key必须是账户id，不能是账户名称，名称可能相同，会造成覆盖
        unordered_map<int, vector<string> > accountid_mails;
        for (auto& [k, v] : mail_accountid) accountid_mails[ds.find_root(v)].emplace_back(k);
        vector<vector<string> > ans;
        for (auto& [k, v] : accountid_mails){
            sort(v.begin(), v.end());
            vector<string> tmp(1, acc[k][0]);
            tmp.insert(tmp.end(), v.begin(), v.end());
            ans.emplace_back(tmp);
        } 
        return ans;
    }
};
```

### [1584. 连接所有点的最小费用](https://leetcode-cn.com/problems/min-cost-to-connect-all-points/)

难度：中等 # 2021.01.19

给你一个`points` 数组，表示 2D 平面上的一些点，其中 `points[i] = [xi, yi]` 。

连接点 `[xi, yi]` 和点 `[xj, yj]` 的费用为它们之间的 **曼哈顿距离** ：`|xi - xj| + |yi - yj|` ，其中 `|val|` 表示 `val` 的绝对值。

请你返回将所有点连接的最小总费用。只有任意两点之间 **有且仅有** 一条简单路径时，才认为所有点都已连接。

**示例 1：**

<table>
    <tl>
        <td>
            <img src="./images/d.png" alt="d"/>
        </td>
        <td>
            <img src="./images/c.png" alt="c"/>
        </td>
    </tl>
</table>

```
输入：points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
输出：20
解释：
我们可以按照上图所示连接所有点得到最小总费用，总费用为 20 。
注意到任意两个点之间只有唯一一条路径互相到达。
```
**示例 2：**
```
输入：points = [[3,12],[-2,5],[-4,1]]
输出：18
```
**示例 3：**
```
输入：points = [[0,0],[1,1],[1,0],[-1,1]]
输出：4
```
**示例 4：**
```
输入：points = [[-1000000,-1000000],[1000000,1000000]]
输出：4000000
```
**示例 5：**
```
输入：points = [[0,0]]
输出：0
```

**提示：**

+ `1 <= points.length <= 1000`
+ `-106 <= xi, yi <= 106`
+ 所有点 `(xi, yi)` 两两不同。

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            return true;
        }
        return false;
    }
};

struct Edge {
    int len, x, y;
    Edge(int len, int x, int y) : len(len), x(x), y(y) {}
};

// static bool cmp(Edge a, Edge b){
//     return a.len < b.len;
// }

class Solution { // 最小生成树 Kruskal
// 首先将完全图中的边全部提取到边集数组中，然后对所有边升序排序；
// 从小到大进行枚举，每次贪心选边加入答案；
// 使用并查集维护连通性，若当前边两端不连通即可选择这条边。
public:
    int distance(vector<vector<int>>& points, int x, int y){
        return abs(points[x][0]-points[y][0]) + abs(points[x][1]-points[y][1]);
    }
    int minCostConnectPoints(vector<vector<int>>& points) {
        int len = points.size();
        Djset ds(len);
        vector<Edge> edges;
        for(int i=0; i<len; i++){
            for(int j=i+1; j<len; j++){
                edges.emplace_back(distance(points, i, j), i, j);
            }
        }
        sort(edges.begin(), edges.end(), [](Edge a, Edge b) -> int { return a.len < b.len; }); // cmp的替代写法
        // sort(edges.begin(), edges.end(), cmp);
        int ans = 0, cnt = 1;
        for(auto& [l, x, y] : edges){
            if(ds.union_root(x, y)){
                ans += l;
                cnt++;
                if(cnt == len){
                    break;
                }
            }
        }
        return ans;
    }
};
```

### [1489. 找到最小生成树里的关键边和伪关键边](https://leetcode-cn.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/)

难度：困难 # 2021.01.21

给你一个 `n` 个点的带权无向连通图，节点编号为 `0` 到 `n-1` ，同时还有一个数组 `edges` ，其中 `edges[i] = [fromi, toi, weighti]` 表示在 `fromi` 和 `toi` 节点之间有一条带权无向边。最小生成树 (MST) 是给定图中边的一个子集，它连接了所有节点且没有环，而且这些边的权值和最小。

请你找到给定图中最小生成树的所有关键边和伪关键边。如果从图中删去某条边，会导致最小生成树的权值和增加，那么我们就说它是一条关键边。伪关键边则是可能会出现在某些最小生成树中但不会出现在所有最小生成树中的边。

请注意，你可以分别以任意顺序返回关键边的下标和伪关键边的下标。

示例 1：

![ex1](./images/ex1.png)

![msts](./images/msts.png)

```
输入：n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
输出：[[0,1],[2,3,4,5]]
解释：上图描述了给定图和所有的最小生成树。

注意到第 0 条边和第 1 条边出现在了所有最小生成树中，所以它们是关键边，我们将这两个下标作为输出的第一个列表。
边 2，3，4 和 5 是所有 MST 的剩余边，所以它们是伪关键边。我们将它们作为输出的第二个列表。
```

**示例 2 ：**

![ex2](./images/ex2.png)

```
输入：n = 4, edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,1]]
输出：[[],[0,1,2,3]]
解释：可以观察到 4 条边都有相同的权值，任选它们中的 3 条可以形成一棵 MST 。所以 4 条边都是伪关键边。
```

**提示：**

+ `2 <= n <= 100`
+ `1 <= edges.length <= min(200, n * (n - 1) / 2)`
+ `edges[i].length == 3`
+ `0 <= fromi < toi < n`
+ `1 <= weighti <= 1000`
+ 所有 `(fromi, toi)` 数对都是互不相同的。

```cpp
/*
这题主要要搞明白以下几点:
1.关键边:所有MST都有的共同边,这个比较好找
2.伪关键边:所有MST包含的边中除了关键边的边,(可不是题目中edges除了关键边的边)
3.'啥都不是边':MST中不可能会加入的边,就是说如果该边加入了,就无法构成MST

先利用克鲁斯卡尔做出一个MST,记录权重,以及构成该MST的边
    然后遍历每一条edges中的边,重新构建一棵MST,对于每一条边有三种情况:
        1.如果是原MST中的边,那么它一定是伪关键边或者关键边的一种,那么如何判断是哪一种呢?
                将其不加入新的MST,如果构成的新MST权重增加了,或者压根就没有连通所有的点,那么该边是关键边
                                如果构成的新的MST权重不增加,则说明该边是伪关键边
        2.如果不是原MST中的边,那么它只能是伪关键边或者'啥都不是边',那么如何判断是哪一种呢?
                将其第一个加入新的MST,如果最终形成的MST权重不增加,那么它就是伪关键边
                                    如果最终形成的MST权重增加了,他就是所谓的'啥都不是边'
*/

class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            return true;
        }
        return false;
    }
};

struct Edge {
    int idx, len, x, y;
    Edge(int idx, int len, int x, int y) : idx(idx), len(len), x(x), y(y) {}
};

class Solution {
public:
    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        Djset ds(n);
        int len = edges.size();
        vector<Edge> edgelist;
        for(int i=0; i<len; i++){
            edgelist.emplace_back(i, edges[i][2], edges[i][0], edges[i][1]);
        }
        sort(edgelist.begin(), edgelist.end(), [](Edge a, Edge b) -> int { return a.len < b.len; });
        int minlen = 0, cnt = 1;
        vector<int> firstkruskal;
        for(auto& [i, l, x, y] : edgelist){
            if(ds.union_root(x, y)){
                firstkruskal.push_back(i);
                minlen += l;
                cnt++;
                if(cnt == n) break;
            }
        }
        vector<vector<int>> ans(2);
        if(cnt != n) return ans; // 不存在最小生成树
        for(int i=0; i<len; i++){
            ds = Djset(n);
            auto p = find(firstkruskal.begin(), firstkruskal.end(), i);
            if(p != firstkruskal.end()){ // 存在于第一次最小生成树的路径中，删掉该边重新生成
                int tmplen = 0;
                cnt = 1;
                for(auto& [idx, l, x, y] : edgelist){
                    if(idx == i) continue;
                    if(ds.union_root(x, y)){
                        tmplen += l;
                        cnt++;
                        if(cnt == n) break;
                    }
                }
                if(cnt != n) ans[0].push_back(i); // 没了这条边都无法生成最小生成树
                else{
                    if(tmplen > minlen) ans[0].push_back(i); // 没了这条边生成的最小生成树路径变长
                    else ans[1].push_back(i);
                }
            } else { // 不存在第一次最小生成树的路径中，算上该边重新生成
                int tmplen = edges[i][2];
                cnt = 2;
                ds.union_root(edges[i][0], edges[i][1]);
                for(auto& [idx, l, x, y] : edgelist){
                    if(idx == i) continue;
                    if(ds.union_root(x, y)){
                        tmplen += l;
                        cnt++;
                        if(cnt == n) break;
                    }
                }
                if(tmplen == minlen) ans[1].push_back(i); // 有这条边也能生成最小生成树
            }
        }
        return ans;
    }
};
```

### [399. 除法求值](https://leetcode-cn.com/problems/evaluate-division/)

难度：中等 # 2021.01.06

给你一个变量对数组 `equations` 和一个实数值数组 `values` 作为已知条件，其中 `equations[i] = [Ai, Bi]` 和 `values[i]` 共同表示等式 `Ai / Bi = values[i]` 。每个 `Ai` 或 `Bi` 是一个表示单个变量的字符串。

另有一些以数组 `queries` 表示的问题，其中 `queries[j] = [Cj, Dj]` 表示第 `j` 个问题，请你根据已知条件找出 `Cj / Dj = ?` 的结果作为答案。

返回 **所有问题的答案** 。如果存在某个无法确定的答案，则用 `-1.0` 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 `-1.0` 替代这个答案。

**注意：**输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

**示例 1：**
```
输入：equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
输出：[6.00000,0.50000,-1.00000,1.00000,-1.00000]
解释：
条件：a / b = 2.0, b / c = 3.0
问题：a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
结果：[6.0, 0.5, -1.0, 1.0, -1.0 ]
```
**示例 2：**
```
输入：equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0], queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
输出：[3.75000,0.40000,5.00000,0.20000]
```
**示例 3：**
```
输入：equations = [["a","b"]], values = [0.5], queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
输出：[0.50000,2.00000,-1.00000,-1.00000]
```
**提示：**

+ `1 <= equations.length <= 20`
+ `equations[i].length == 2`
+ `1 <= Ai.length, Bi.length <= 5`
+ `values.length == equations.length`
+ `0.0 < values[i] <= 20.0`
+ `1 <= queries.length <= 20`
+ `queries[i].length == 2`
+ `1 <= Cj.length, Dj.length <= 5`
+ `Ai, Bi, Cj, Dj` 由小写英文字母与数字组成

```cpp
class Solution {
public:
    double dfs(int s, int t, vector<vector<double> > &mat, int cnt, vector<vector<int> > &v){
        v[s][t] = 1;
        v[t][s] = 1;
        // cout<<"s: "<<s<<" t: "<<t<<endl;
        if(s == t)
            return 1;
        if(fabs(mat[s][t] - 0) > 1e-5)
            return mat[s][t];
        else{
            for(int i=0; i<cnt; i++){
                if(i != s && v[i][t] == 0 && fabs(mat[s][i] - 0) > 1e-5){
                    double a = dfs(i, t, mat, cnt, v);
                    if(fabs(a + 1) < 1e-5)
                        continue;
                    a *= mat[s][i];
                    // cout<<a<<endl;
                    return a;
                }
            }
        }
        return -1;
    }
    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        int n = equations.size();
        vector<vector<double> >  mat(n*2, vector<double>(n*2, 0)); // 邻接矩阵
        map<string, int> m; // 为每个变量编号
        int cnt = 0;
        for(int i=0; i<n; i++){
            if(m.count(equations[i][0]) == 0){
                m[equations[i][0]] = cnt;
                cnt++;
            }
            if(m.count(equations[i][1]) == 0){
                m[equations[i][1]] = cnt;
                cnt++;
            }
            mat[m[equations[i][0]]][m[equations[i][1]]] = values[i];
            mat[m[equations[i][1]]][m[equations[i][0]]] = 1 / values[i];
        }
        vector<vector<int> > v(cnt, vector<int>(cnt, 0));
        vector<double> ans;
        int q = queries.size();
        for(int i=0; i<q; i++){
            // cout<<"case: "<<i+1<<endl;
            if(m.count(queries[i][0]) * m.count(queries[i][1]) == 0){ // 未出现的变量
                ans.push_back(-1);
                continue;
            }
            double a = dfs(m[queries[i][0]], m[queries[i][1]], mat, cnt, v);
            ans.push_back(a);
            v = vector<vector<int> >(cnt, vector<int>(cnt, 0));
        }
        return ans;
    }
};
```



### [547. 省份数量](https://leetcode-cn.com/problems/number-of-provinces/)

难度：中等 # 2021.01.07

有 `n` 个城市，其中一些彼此相连，另一些没有相连。如果城市 `a` 与城市 `b` 直接相连，且城市 `b` 与城市 `c` 直接相连，那么城市 `a` 与城市 `c` 间接相连。

**省份** 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 `n x n` 的矩阵 `isConnected` ，其中 `isConnected[i][j] = 1` 表示第 `i` 个城市和第 `j` 个城市直接相连，而 `isConnected[i][j] = 0` 表示二者不直接相连。

返回矩阵中 **省份** 的数量。

 **示例 1：**

![graph1](./images/graph1.jpg)

```
输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
```

**示例 2：**

![graph2](./images/graph2.jpg)

```
输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
输出：3
```

**提示：**

+ `1 <= n <= 200`
+ `n == isConnected.length`
+ `n == isConnected[i].length`
+ `isConnected[i][j]` 为 `1` 或 `0`
+ `isConnected[i][i] == 1`
+ `isConnected[i][j] == isConnected[j][i]`

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            return true;
        }
        return false;
    }
};
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int len = isConnected.size();
        Djset ds(len);
        int ans = len;
        for(int i=0; i<len; i++){
            for(int j=0; j<len; j++){
                if(isConnected[i][j] == 1){
                    if(ds.union_root(i, j)) ans--;
                }
            }
        }
        return ans;
    }
};

class Solution { // dfs
public:
    void dfs(vector<vector<int>>& isConnected, vector<int>& v, int c, int len){
        v[c] = 1;
        for(int i=0; i<len; i++){
            if(v[i] == 0 && isConnected[c][i] == 1)
                dfs(isConnected, v, i, len);
        }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        int len = isConnected.size();
        vector<int> v(len, 0);
        int cnt = 0;
        for(int i=0; i<len; i++){
            if(v[i] == 0){
                cnt++;
                dfs(isConnected, v, i, len);
            }
        }
        return cnt;
    }
};
```

### [947. 移除最多的同行或同列石头](https://leetcode-cn.com/problems/most-stones-removed-with-same-row-or-column/)

难度：中等 # 2021.01.15

`n` 块石头放置在二维平面中的一些整数坐标点上。每个坐标点上最多只能有一块石头。

如果一块石头的 **同行或者同列** 上有其他石头存在，那么就可以移除这块石头。

给你一个长度为 `n` 的数组 `stones` ，其中 `stones[i] = [xi, yi]` 表示第 `i` 块石头的位置，返回 **可以移除的石子** 的最大数量。

**示例 1：**
```
输入：stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
输出：5
解释：一种移除 5 块石头的方法如下所示：
1. 移除石头 [2,2] ，因为它和 [2,1] 同行。
2. 移除石头 [2,1] ，因为它和 [0,1] 同列。
3. 移除石头 [1,2] ，因为它和 [1,0] 同行。
4. 移除石头 [1,0] ，因为它和 [0,0] 同列。
5. 移除石头 [0,1] ，因为它和 [0,0] 同行。
石头 [0,0] 不能移除，因为它没有与另一块石头同行/列。
```
**示例 2：**
```
输入：stones = [[0,0],[0,2],[1,1],[2,0],[2,2]]
输出：3
解释：一种移除 3 块石头的方法如下所示：
1. 移除石头 [2,2] ，因为它和 [2,0] 同行。
2. 移除石头 [2,0] ，因为它和 [0,0] 同列。
3. 移除石头 [0,2] ，因为它和 [0,0] 同行。
石头 [0,0] 和 [1,1] 不能移除，因为它们没有与另一块石头同行/列。
```
**示例 3：**

```
输入：stones = [[0,0]]
输出：0
解释：[0,0] 是平面上唯一一块石头，所以不可以移除它。
```

**提示：**

+ `1 <= stones.length <= 1000`
+ `0 <= xi, yi <= 104`
+ 不会有两块石头放在同一个坐标点上

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            return true;
        }
        return false;
    }
};

class Solution {
public:
    int removeStones(vector<vector<int>> &stones) { // 用横纵坐标归为同集
        int len = stones.size();
        Djset ds(20002); // 坐标区间
        for(int i=0; i<len; i++){
            ds.union_root(stones[i][0], stones[i][1]+10001); // 把横纵坐标的空间分开
        }
        set<int> s;
        for(int i=0; i<len; i++){
            s.insert(ds.find_root(stones[i][0]));
            s.insert(ds.find_root(stones[i][1]+10001));
        }
        return len - s.size();
    }
};
```

### [959. 由斜杠划分区域](https://leetcode-cn.com/problems/regions-cut-by-slashes/)

难度：中等 # 2021.01.25

在由 1 x 1 方格组成的 N x N 网格 `grid` 中，每个 1 x 1 方块由 `/`、`\` 或空格构成。这些字符会将方块划分为一些共边的区域。

（请注意，反斜杠字符是转义的，因此 `\` 用 `"\\"` 表示。）。

返回区域的数目。

**示例 1：**
```
输入：
[
  " /",
  "/ "
]
输出：2
解释：2x2 网格如下：
```
![1](./images/1.png)

**示例 2：**
```
输入：
[
  " /",
  "  "
]
输出：1
解释：2x2 网格如下：
```
![2](./images/2.png)

**示例 3：**
```
输入：
[
  "\\/",
  "/\\"
]
输出：4
解释：（回想一下，因为 \ 字符是转义的，所以 "\\/" 表示 \/，而 "/\\" 表示 /\。）
2x2 网格如下：
```
![3](./images/3.png)

**示例 4：**
```
输入：
[
  "/\\",
  "\\/"
]
输出：5
解释：（回想一下，因为 \ 字符是转义的，所以 "/\\" 表示 /\，而 "\\/" 表示 \/。）
2x2 网格如下：
```
![4](./images/4.png)

**示例 5：**
```
输入：
[
  "//",
  "/ "
]
输出：3
解释：2x2 网格如下：
```
![5](./images/5.png)

**提示：**

1. `1 <= grid.length == grid[0].length <= 30`
2. `grid[i][j]` 是 `'/'`、`'\'`、或 `' '`。

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    int cnt;
    Djset(int n): parent(vector<int>(n)), cnt(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int get_cnt() {
        return cnt;
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            cnt--;
            return true;
        }
        return false;
    }
};

class Solution {
public:
    int regionsBySlashes(vector<string>& grid) {
        int N = grid.size();
        Djset ds(4*N*N); // 二维映射到一维，每个方格从上面开始顺时针分成4块
        for(int i=0; i<N; i++){
            for(int j=0; j<N; j++){
                // 方格内合并
                int idx = 4 * (i * N + j);
                if(grid[i][j] == '/'){
                    ds.union_root(idx, idx+3);
                    ds.union_root(idx+1, idx+2);
                } else if(grid[i][j] == '\\'){
                    ds.union_root(idx+0, idx+1);
                    ds.union_root(idx+2, idx+3);
                } else {
                    ds.union_root(idx, idx+1);
                    ds.union_root(idx+1, idx+2);
                    ds.union_root(idx+2, idx+3);
                }
                // 方格间合并
                int idx_r = idx + 4 + 3; // 右侧方格中靠左侧的小块
                if(j + 1 < N) ds.union_root(idx+1, idx_r);
                int idx_b = idx + 4 * N; // 下侧方格中靠上册的小块
                if(i + 1 < N) ds.union_root(idx+2, idx_b);
            }
        }
        return ds.get_cnt();
    }
};
```

### [1579. 保证图可完全遍历](https://leetcode-cn.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/)

难度：困难 # 2021.01.27

Alice 和 Bob 共有一个无向图，其中包含 n 个节点和 3 种类型的边：

- 类型 1：只能由 Alice 遍历。
- 类型 2：只能由 Bob 遍历。
- 类型 3：Alice 和 Bob 都可以遍历。

给你一个数组 `edges` ，其中 `edges[i] = [typei, ui, vi]` 表示节点 `ui` 和 `vi` 之间存在类型为 `typei` 的双向边。请你在保证图仍能够被 Alice和 Bob 完全遍历的前提下，找出可以删除的最大边数。如果从任何节点开始，Alice 和 Bob 都可以到达所有其他节点，则认为图是可以完全遍历的。

返回可以删除的最大边数，如果 Alice 和 Bob 无法完全遍历图，则返回 -1 。

**示例 1：**

![5510ex1](./images/5510ex1.png)
```
输入：n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,2],[2,3,4]]
输出：2
解释：如果删除 [1,1,2] 和 [1,1,3] 这两条边，Alice 和 Bob 仍然可以完全遍历这个图。再删除任何其他的边都无法保证图可以完全遍历。所以可以删除的最大边数是 2 。
```
**示例 2：**

![5510ex2](./images/5510ex2.png)
```
输入：n = 4, edges = [[3,1,2],[3,2,3],[1,1,4],[2,1,4]]
输出：0
解释：注意，删除任何一条边都会使 Alice 和 Bob 无法完全遍历这个图。
```
**示例 3：**

![5510ex3](./images/5510ex3.png)
```
输入：n = 4, edges = [[3,2,3],[1,1,2],[2,3,4]]
输出：-1
解释：在当前图中，Alice 无法从其他节点到达节点 4 。类似地，Bob 也不能达到节点 1 。因此，图无法完全遍历。
```

**提示：**

+ `1 <= n <= 10^5`
+ `1 <= edges.length <= min(10^5, 3 * n * (n-1) / 2)`
+ `edges[i].length == 3`
+ `1 <= edges[i][0] <= 3`
+ `1 <= edges[i][1] < edges[i][2] <= n`
+ 所有元组 `(typei, ui, vi)` 互不相同

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    int cnt;
    Djset(int n): parent(vector<int>(n)), cnt(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int get_cnt() {
        return cnt;
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            cnt--;
            return true;
        }
        return false;
    }
};

class Solution {
public:
    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        vector<pair<int, int>> edges1;
        vector<pair<int, int>> edges2;        
        vector<pair<int, int>> edges3;
        int len = edges.size();
        for(auto& edge: edges){
            if(edge[0] == 1) edges1.push_back(pair<int, int>(edge[1]-1, edge[2]-1));
            else if(edge[0] == 2) edges2.push_back(pair<int, int>(edge[1]-1, edge[2]-1));
            else edges3.push_back(pair<int, int>(edge[1]-1, edge[2]-1));
        }
        int ans = 0;
        Djset ds3(n);
        for(auto& edge: edges3){
            if(!ds3.union_root(edge.first, edge.second)) ans++;
        }
        Djset ds1 = ds3;
        for(auto& edge: edges1){
            if(!ds1.union_root(edge.first, edge.second)) ans++;
        }
        if(ds1.get_cnt() != 1) return -1;
        Djset ds2 = ds3;
        for(auto& edge: edges2){
            if(!ds2.union_root(edge.first, edge.second)) ans++;
        }
        if(ds2.get_cnt() != 1) return -1;
        return ans;
    }
};
```

### [1631. 最小体力消耗路径](https://leetcode-cn.com/problems/path-with-minimum-effort/)

难度：中等 # 2021.01.29

你准备参加一场远足活动。给你一个二维 `rows x columns` 的地图 `heights` ，其中 `heights[row][col]` 表示格子 `(row, col)` 的高度。一开始你在最左上角的格子 `(0, 0)` ，且你希望去最右下角的格子 `(rows-1, columns-1)` （注意下标从 **0** 开始编号）。你每次可以往 **上，下，左，右** 四个方向之一移动，你想要找到耗费 **体力** 最小的一条路径。

一条路径耗费的 **体力值** 是路径上相邻格子之间 **高度差绝对值** 的 **最大值** 决定的。

请你返回从左上角走到右下角的最小 **体力消耗值** 。

**示例 1：**
<img src="./images/1631ex1.png" alt="1631ex1" style="zoom:50%;" />

```
输入：heights = [[1,2,2],[3,8,2],[5,3,5]]
输出：2
解释：路径 [1,3,5,3,5] 连续格子的差值绝对值最大为 2 。
这条路径比路径 [1,2,2,2,5] 更优，因为另一条路径差值最大值为 3 。
```
**示例 2：**
<img src="./images/1631ex2.png" alt="1631ex2" style="zoom:50%;" />

```
输入：heights = [[1,2,3],[3,8,4],[5,3,5]]
输出：1
解释：路径 [1,2,3,4,5] 的相邻格子差值绝对值最大为 1 ，比路径 [1,3,5,3,5] 更优。
```
**示例 3：**
<img src="./images/1631ex3.png" alt="1631ex3" style="zoom:50%;" />

```
输入：heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
输出：0
解释：上图所示路径不需要消耗任何体力。
```
**提示：**

+ `rows == heights.length`
+ `columns == heights[i].length`
+ `1 <= rows, columns <= 100`
+ `1 <= heights[i][j] <= 106`

```cpp
class Solution { // 二分 + dfs 超时
public:
    bool dfs(vector<vector<int>>& heights, int& r, int& c, vector<vector<int>>& v, int& gap, int x, int y){
        if(x == r-1 && y == c-1) return true;
        v[x][y] = 1;
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, -1, 0, 1};
        bool ans = false;
        for(int i=0; i<4; i++){
            int xx = x + dx[i];
            int yy = y + dy[i];
            if(xx>=0 && xx<r && yy>=0 && yy<c && v[xx][yy]==0 && abs(heights[x][y]-heights[xx][yy])<=gap){
                if(dfs(heights, r, c, v, gap, xx, yy)){
                    ans = true;
                    break;
                }
            }
        }
        v[x][y] = 0;
        return ans;
    }
    int minimumEffortPath(vector<vector<int>>& heights) {
        int r = heights.size();
        int c = heights[0].size();
        vector<vector<int>> v(r, vector<int>(c, 0));
        int mi = INT_MAX, ma = INT_MIN;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                mi = min(mi, heights[i][j]);
                ma = max(ma, heights[i][j]);
            }
        }
        int max_gap = ma - mi;
        int left = 0, right = max_gap, mid;
        while(left <= right){
            mid = left + (right-left)/2;
            if(dfs(heights, r, c, v, mid, 0, 0)) right = mid - 1;
            else left = mid + 1;
        }
        return left;
    }
};

// Kruskal + 并查集
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            return true;
        }
        return false;
    }
};
struct Edge {
    int start, end, len;
    Edge(int start, int end, int len) : start(start), end(end), len(len) {}
};
class Solution {
public:
    int minimumEffortPath(vector<vector<int>>& heights) {
        int r = heights.size();
        int c = heights[0].size();
        int minCost = 0;
        vector<Edge> edges;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(i + 1 < r)
                    edges.push_back({i*c+j, (i+1)*c+j, abs(heights[i+1][j] - heights[i][j])});
                if(j + 1 < c)
                    edges.push_back({i*c+j, i*c+j+1, abs(heights[i][j+1] - heights[i][j])});
            }
        }
        sort(edges.begin(), edges.end(), [](auto& a, auto& b) -> int { return a.len < b.len; });
        Djset ds(r * c);
        for (auto& e :edges) {
            minCost = e.len;
            ds.union_root(e.start, e.end);   
            if(ds.find_root(0) == ds.find_root(r*c-1)) break;
        }
        return minCost;
    }
};
```

### [778. 水位上升的泳池中游泳](https://leetcode-cn.com/problems/swim-in-rising-water/)

难度：困难 # 2021.01.30

在一个 N x N 的坐标方格 `grid` 中，每一个方格的值 `grid[i][j]` 表示在位置 `(i,j)` 的平台高度。

现在开始下雨了。当时间为 `t` 时，此时雨水导致水池中任意位置的水位为 `t` 。你可以从一个平台游向四周相邻的任意一个平台，但是前提是此时水位必须同时淹没这两个平台。假定你可以瞬间移动无限距离，也就是默认在方格内部游动是不耗时的。当然，在你游泳的时候你必须待在坐标方格里面。

你从坐标方格的左上平台 (0，0) 出发。最少耗时多久你才能到达坐标方格的右下平台 `(N-1, N-1)`？

**示例 1：**
```
输入: [[0,2],[1,3]]
输出: 3
解释:
时间为0时，你位于坐标方格的位置为 (0, 0)。
此时你不能游向任意方向，因为四个相邻方向平台的高度都大于当前时间为 0 时的水位。

等时间到达 3 时，你才可以游向平台 (1, 1). 因为此时的水位是 3，坐标方格中的平台没有比水位 3 更高的，所以你可以游向坐标方格中的任意位置
```
**示例2：**
```
输入: [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
输出: 16
解释:
 0  1  2  3  4
24 23 22 21  5
12 13 14 15 16
11 17 18 19 20
10  9  8  7  6

最终的路线用加粗进行了标记。
我们必须等到时间为 16，此时才能保证平台 (0, 0) 和 (4, 4) 是连通的
```

**提示：**

+ `2 <= N <= 50`.
+ `grid[i][j]` 是 `[0, ..., N*N - 1]` 的排列。

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    Djset(int n): parent(vector<int>(n)) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            return true;
        }
        return false;
    }
};
struct Edge {
    int start, end, len;
    Edge(int start, int end, int len) : start(start), end(end), len(len) {}
};
class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int r = grid.size();
        int c = grid[0].size();
        int minCost = 0;
        vector<Edge> edges;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(i + 1 < r)
                    edges.push_back({i*c+j, (i+1)*c+j, max(grid[i+1][j], grid[i][j])});
                if(j + 1 < c)
                    edges.push_back({i*c+j, i*c+j+1, max(grid[i][j+1], grid[i][j])});
            }
        }
        sort(edges.begin(), edges.end(), [](auto& a, auto& b) -> int { return a.len < b.len; });
        Djset ds(r * c);
        for (auto& e :edges) {
            minCost = e.len;
            ds.union_root(e.start, e.end);   
            if(ds.find_root(0) == ds.find_root(r*c-1)) break;
        }
        return minCost;
    }
};
```

### [839. 相似字符串组](https://leetcode-cn.com/problems/similar-string-groups/)

难度：困难 # 2021.01.31

如果交换字符串 `X` 中的两个不同位置的字母，使得它和字符串 `Y` 相等，那么称 `X` 和 `Y` 两个字符串相似。如果这两个字符串本身是相等的，那它们也是相似的。

例如，`"tars"` 和 `"rats"` 是相似的 (交换 `0` 与 `2` 的位置)； `"rats"` 和 `"arts"` 也是相似的，但是 `"star"` 不与 `"tars"`，`"rats"`，或 `"arts"` 相似。

总之，它们通过相似性形成了两个关联组：`{"tars", "rats", "arts"}` 和 `{"star"}`。注意，`"tars"` 和 `"arts"` 是在同一组中，即使它们并不相似。形式上，对每个组而言，要确定一个单词在组中，只需要这个词和该组中至少一个单词相似。

给你一个字符串列表 `strs`。列表中的每个字符串都是 `strs` 中其它所有字符串的一个字母异位词。请问 `strs` 中有多少个相似字符串组？

**示例 1：**
```
输入：strs = ["tars","rats","arts","star"]
输出：2
```
**示例 2：**
```
输入：strs = ["omv","ovm"]
输出：1
```
**提示：**

+ `1 <= strs.length <= 100`
+ `1 <= strs[i].length <= 1000`
+ `sum(strs[i].length) <= 2 * 104`
+ `strs[i]` 只包含小写字母。
+ `strs` 中的所有单词都具有相同的长度，且是彼此的字母异位词。

**备注：**

      字母异位词（anagram），一种把某个字符串的字母的位置（顺序）加以改换所形成的新词。

```cpp
class Djset { // Union-find disjoint set 并查集模板
public:
    vector<int> parent;
    int cnt;
    Djset(int n): parent(vector<int>(n)), cnt(n) {
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int get_cnt() {
        return cnt;
    }

    int find_root(int x) {
        if (x != parent[x]) {
            parent[x] = find_root(parent[x]);
        }
        return parent[x];
    }

    bool union_root(int x, int y) {
        int root_x = find_root(x);
        int root_y = find_root(y);
        if (root_x != root_y) {
            parent[root_x] = root_y;
            cnt--;
            return true;
        }
        return false;
    }
};
class Solution {
public:
    bool isSimilar(string a, string b){
        int len = a.size(), blen = b.size();
        if(len != blen) return false;
        int cnt = 0;
        char ac[2], bc[2];
        for(int i=0; i<len; i++){
            if(a[i] == b[i]) continue;
            if(cnt < 2){
                ac[cnt] = a[i];
                bc[cnt] = b[i];
                cnt++;
            } else {
                return false;
            }
        }
        if(cnt == 0) return true;
        if(cnt == 2 && ac[0] == bc[1] && ac[1] == bc[0]) return true;
        return false;
    }
    int numSimilarGroups(vector<string>& strs) {
        int len = strs.size();
        Djset ds(len);
        for(int i=0; i<len; i++){
            for(int j=0; j<len; j++){
                if(isSimilar(strs[i], strs[j])) ds.union_root(i, j);
            }
        }
        return ds.get_cnt();
    }
};
```



## 二分

### [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

难度：简单 # 2020.11.03

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

**示例 1：**
```
输入: [1,3,5,6], 5
输出: 2
```
**示例 2：**
```
输入: [1,3,5,6], 2
输出: 1
```
**示例 3：**
```
输入: [1,3,5,6], 7
输出: 4
```
**示例 4：**
```
输入: [1,3,5,6], 0
输出: 0
```

```cpp
class Solution { // 二分
public:
    int searchInsert(vector<int>& nums, int target) {
        int len = nums.size();
        int l=0, r=len-1, mid; // 左闭右闭
        while(l <= r){ // 小于等于
            mid = l + (r-l)/2; // 别用 (l+r)/2
            if(nums[mid] == target) return mid;
            else if(nums[mid] > target) r = mid-1;
            else l = mid+1;
        }
        return l; // 返回l
    }
};

class Solution { // nb就完事了
public:
    int searchInsert(vector<int>& nums, int target) {
        return lower_bound(nums.begin(), nums.end(), target) - nums.begin();
    }
};
```

### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

难度：中等 # 2020.11.03

给定一个按照升序排列的整数数组 `nums`，和一个目标值 `target`。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

进阶：

+ 你可以设计并实现时间复杂度为 `O(log n)` 的算法解决此问题吗？

**示例 1：**
```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```
**示例 2：**
```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```
**示例 3：**
```
输入：nums = [], target = 0
输出：[-1,-1]
```
**提示：**

+ `0 <= nums.length <= 105`
+ `-109 <= nums[i] <= 109`
+ `nums` 是一个非递减数组
+ `-109 <= target <= 109`

```cpp
class Solution { // 二分
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int len = nums.size();
        int l=0, r=len-1, lower, upper;
        bool exist = false;
        while(l <= r){ // 算下界
            lower = l + (r-l)/2;
            if(nums[lower] < target) l = lower + 1;
            else{
                r = lower - 1;
                if(nums[lower] == target) exist = true; // 标志是否存在
            }
        }
        if(!exist) return vector<int>{-1, -1}; // 不存在直接返回
        lower = l;
        l=0, r=len-1;
        while(l <= r){ // 算上界
            upper = l + (r-l)/2;
            if(nums[upper] <= target) l = upper + 1; // 这个等于号用来控制上界还是下界
            else r = upper - 1;
        }
        upper = r;
        return vector<int>{lower, upper};
    }
};

class Solution { // nb就完事了
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int len = nums.size();
        if(len == 0) return vector<int>{-1, -1};
        int lower = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
        int upper = upper_bound(nums.begin(), nums.end(), target) - nums.begin();
        // cout<<lower<<" "<<upper<<endl;
        if(lower < len && nums[lower] == target) return vector<int>{lower, upper-1};
        return vector<int>{-1, -1};
    }
};
```

### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

难度：中等 # 2020.11.04

升序排列的整数数组 `nums` 在预先未知的某个点上进行了旋转（例如， `[0,1,2,4,5,6,7]` 经旋转后可能变为 `[4,5,6,7,0,1,2]` ）。

请你在数组中搜索 `target` ，如果数组中存在这个目标值，则返回它的索引，否则返回 `-1` 。

**示例 1：**
```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```
**示例 2：**
```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```
**示例 3：**
```
输入：nums = [1], target = 0
输出：-1
```
**提示：**

+ `1 <= nums.length <= 5000`
+ `-10^4 <= nums[i] <= 10^4`
+ `nums` 中的每个值都 **独一无二**
+ `nums` 肯定会在某个点上旋转
+ `-10^4 <= target <= 10^4`

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int len = nums.size(), pivot;
        if(len == 1) return nums[0] == target ? 0 : -1;
        for(pivot=0; pivot<len-1; pivot++){
            if(nums[pivot] > nums[pivot+1]) break;
        }
        int l, r, mid;
        if(pivot == len-1){ // 没有旋转，正常二分
            l = 0;
            r = len - 1;
        }
        else if(nums[pivot] < target) return -1; // target比最大的大
        else if(nums[0] <= target){ // 左侧段
            l = 0;
            r = pivot;
        }
        else{ // 右侧段
            l = pivot + 1;
            r = len-1;
        }
        while(l <= r){
            mid = l + (r-l)/2;
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return -1;
    }
};
```

### [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

难度：中等 # 2020.11.05

峰值元素是指其值大于左右相邻值的元素。

给定一个输入数组 `nums`，其中 `nums[i] ≠ nums[i+1]`，找到峰值元素并返回其索引。

数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。

你可以假设 `nums[-1] = nums[n] = -∞`。

**示例 1：**
```
输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。
```
**示例 2：**
```
输入: nums = [1,2,1,3,5,6,4]
输出: 1 或 5 
解释: 你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。
```
**说明：**

你的解法应该是 O(logN) 时间复杂度的。

```cpp
class Solution { // O(N)
public:
    int findPeakElement(vector<int>& nums) {
        int len = nums.size();
        if(len == 1) return 0;
        if(nums[0] > nums[1]) return 0;
        if(nums[len-1] > nums[len-2]) return len-1;
        for(int i=1; i<len-1; i++){
            if(nums[i] > nums[i-1] && nums[i] > nums[i+1]) return i;
        }
        return -1;
    }
};
```

### [74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

难度：中等 # 远古

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

+ 每行中的整数从左到右按升序排列。
+ 每行的第一个整数大于前一行的最后一个整数。

**示例 1：**

![mat](./images/mat.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 3
输出：true
```

**示例 2：**

![mat2](./images/mat2.jpg)

```
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 13
输出：false
```

**示例 3：**

```
输入：matrix = [], target = 0
输出：false
```

**提示：**

`m == matrix.length`
`n == matrix[i].length`
`0 <= m, n <= 100`
`-104 <= matrix[i][j], target <= 104`

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        if(m == 0)
            return false;
        int n = matrix[0].size();
        if(n == 0)
            return false;
        int l = 0;
        int h = m * n - 1;
        int mid;
        while(l <= h){
            mid = l + (h - l) / 2;
            int r = mid / n;
            int c = mid % n;
            if(target == matrix[r][c])
                return true;
            else if(target > matrix[r][c]){
                l = mid + 1;
            }
            else if(target < matrix[r][c]){
                h = mid - 1;
            }
        }
        return false;
    }
};
```

### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

难度：中等 # 远古

编写一个高效的算法来搜索 `m x n` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

+ 每行的元素从左到右升序排列。
+ 每列的元素从上到下升序排列。

**示例 1：**

![searchgrid2](./images/searchgrid2.jpg)

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
```
**示例 2：**

![searchgrid](./images/searchgrid.jpg)

```
输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
```

**提示：**

+ `m == matrix.length`
+ `n == matrix[i].length`
+ `1 <= n, m <= 300`
+ `-109 <= matix[i][j] <= 109`
+ 每行的所有元素从左到右升序排列
+ 每列的所有元素从上到下升序排列
+ `-109 <= target <= 109`

```cpp
class Solution { // 愚蠢的递归做法
public:
    bool search(vector<vector<int>>& matrix, int& m, int& n, int lr, int lc, int hr, int hc, int& t){
        //cout<<lr<<lc<<hr<<hc<<endl;
        if(lr < 0 || lc < 0 || hr >= m || hc >= n || lr > hr || lc > hc || t < matrix[lr][lc] || t > matrix[hr][hc])
            return false;
        int midr = lr + (hr - lr) / 2;
        int midc = lc + (hc - lc) / 2;
        if(matrix[midr][midc] == t){
            //cout<<1<<endl;
            return true;
        }
        else if(matrix[midr][midc] > t){
            //cout<<2<<endl;
            return search(matrix, m, n, midr, lc, hr, midc-1, t) || search(matrix, m, n, lr, midc, midr-1, hc, t) || search(matrix, m, n, lr, lc, midr-1, midc-1, t);
        }
        else{
            //cout<<3<<endl;
            return search(matrix, m, n, midr+1, lc, hr, midc, t) || search(matrix, m, n, lr, midc+1, midr, hc, t) || search(matrix, m, n, midr+1, midc+1, hr, hc, t);
        }
        return false;
    }
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        if(m == 0)
            return false;
        int n = matrix[0].size();
        if(n == 0)
            return false;
        if(target >= matrix[0][0] && target <= matrix[m-1][n-1])
            return search(matrix, m, n, 0, 0, m-1, n-1, target);
        return false;
    }
};
```

### [1552. 两球之间的磁力](https://leetcode-cn.com/problems/magnetic-force-between-two-balls/)

难度：中等 # 2020.11.07

在代号为 C-137 的地球上，Rick 发现如果他将两个球放在他新发明的篮子里，它们之间会形成特殊形式的磁力。Rick 有 `n` 个空的篮子，第 i 个篮子的位置在 `position[i]` ，Morty 想把 `m` 个球放到这些篮子里，使得任意两球间 最小磁力 最大。

已知两个球如果分别位于 `x` 和 `y` ，那么它们之间的磁力为 `|x - y|` 。

给你一个整数数组 `position` 和一个整数 `m` ，请你返回最大化的最小磁力。

**示例 1：**

![q3v1](./images/q3v1.jpg)
```
输入：position = [1,2,3,4,7], m = 3
输出：3
解释：将 3 个球分别放入位于 1，4 和 7 的三个篮子，两球间的磁力分别为 [3, 3, 6]。最小磁力为 3 。我们没办法让最小磁力大于 3 。
```
**示例 2：**
```
输入：position = [5,4,3,2,1,1000000000], m = 2
输出：999999999
解释：我们使用位于 1 和 1000000000 的篮子时最小磁力最大。
```

**提示：**

+ `n == position.length`
+ `2 <= n <= 10^5`
+ `1 <= position[i] <= 10^9`
+ 所有 `position` 中的整数 互不相同 。
+ `2 <= m <= position.length`

```cpp
class Solution {
public:
    bool check(int x, vector<int>& position, int m) {
        int pre = position[0], cnt = 1;
        for (int i = 1; i < position.size(); ++i) {
            if (position[i] - pre >= x) {
                pre = position[i];
                cnt += 1;
            }
            if(cnt >= m) return true;
        }
        return false;
    }
    int maxDistance(vector<int>& position, int m) { // 二分查找满足的最小磁力
        sort(position.begin(), position.end());
        int left=1, right=(position.back()-position[0]) / (m-1), ans=0;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (check(mid, position, m)) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }
};
```

### [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

难度：中等 # 2021.04.07

已知存在一个按非降序排列的整数数组 `nums` ，数组中的值不必互不相同。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转** ，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,4,4,5,6,6,7]` 在下标 `5` 处经旋转后可能变为 `[4,5,6,6,7,0,1,2,4,4]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 `nums` 中存在这个目标值 `target` ，则返回 `true` ，否则返回 `false` 。

**示例 1：**
```
输入：nums = [2,5,6,0,0,1,2], target = 0
输出：true
```
**示例 2：**
```
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false
```

**提示：**

+ `1 <= nums.length <= 5000`
+ `-10^4 <= nums[i] <= 10^4`
+ 题目数据保证 `nums` 在预先未知的某个下标上进行了旋转
+ `-10^4 <= target <= 10^4`

**进阶：**

+ 这是 搜索旋转排序数组 的延伸题目，本题中的 `nums`  可能包含重复元素。
+ 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么？

```cpp
class Solution {
public:
    bool search(vector<int>& nums, int target) {
        int len = nums.size(), pivot;
        if(len == 1) return nums[0] == target ? true : false;
        for(pivot=0; pivot<len-1; pivot++){
            if(nums[pivot] > nums[pivot+1]) break;
        }
        int l, r, mid;
        if(pivot == len-1){ // 没有旋转，正常二分
            l = 0;
            r = len - 1;
        }
        else if(nums[pivot] < target) return false; // target比最大的大
        else if(nums[0] <= target){ // 左侧段
            l = 0;
            r = pivot;
        }
        else{ // 右侧段
            l = pivot + 1;
            r = len-1;
        }
        while(l <= r){
            mid = l + (r-l)/2;
            if(nums[mid] == target) return true;
            else if(nums[mid] < target) l = mid + 1;
            else r = mid - 1;
        }
        return false;
    }
};
```



## 深度优先搜索

### [面试题 04.04. 检查平衡性](https://leetcode-cn.com/problems/check-balance-lcci/)

难度：简单 # 2020.11.07

实现一个函数，检查二叉树是否平衡。在这个问题中，平衡树的定义如下：任意一个节点，其两棵子树的高度差不超过 1。


**示例 1：**
```
给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
返回 true 。
```
**示例 2：**
```
给定二叉树 [1,2,2,3,3,null,null,4,4]
      1
     / \
    2   2
   / \
  3   3
 / \
4   4
返回 false 。
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int depth(TreeNode* root, bool& ans){
        if(ans == false) return -1;
        if(!root) return 0;
        int llen = depth(root->left, ans);
        int rlen = depth(root->right, ans);
        if(abs(llen - rlen) > 1) ans = false;
        return max(llen, rlen) + 1;

    }
    bool isBalanced(TreeNode* root) {
        bool ans = true;
        depth(root, ans);
        return ans;
    }
};
```

### [257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)

难度：简单 # 2020.11.09

给定一个二叉树，返回所有从根节点到叶子节点的路径。

**说明：**叶子节点是指没有子节点的节点。

**示例：**
```
输入:
   1
 /   \
2     3
 \
  5
输出: ["1->2->5", "1->3"]
解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    string convert(vector<int> path){
        int len = path.size();
        string ans;
        for(int i=0; i<len-1; i++){
            ans += to_string(path[i]) + "->";
        }
        ans += to_string(path[len-1]);
        return ans;
    }
    void dfs(TreeNode* root, vector<int>& path, vector<string>& ans){
        if(!root->left && !root->right){
            ans.push_back(convert(path));
            return;
        }
        TreeNode *rl = root->left, *rr = root->right;
        if(rl){
            path.push_back(rl->val);
            dfs(rl, path, ans);
            path.pop_back();
        }
        if(rr){
            path.push_back(rr->val);
            dfs(rr, path, ans);
            path.pop_back();
        }
        return;
    }
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> ans;
        vector<int> path;
        if(root) path.push_back(root->val);
        else return ans;
        dfs(root, path, ans);
        return ans;
    }
};
```

### [面试题 16.19. 水域大小](https://leetcode-cn.com/problems/pond-sizes-lcci/)

难度：中等 # 2020.11.16

你有一个用于表示一片土地的整数矩阵land，该矩阵中每个点的值代表对应地点的海拔高度。若值为0则表示水域。由垂直、水平或对角连接的水域为池塘。池塘的大小是指相连接的水域的个数。编写一个方法来计算矩阵中所有池塘的大小，返回值需要从小到大排序。

**示例：

```
输入：
[
  [0,2,1,0],
  [0,1,0,1],
  [1,1,0,1],
  [0,1,0,1]
]
输出： [1,2,4]
```

提示：

+ `0 < len(land) <= 1000`
+ `0 < len(land[i]) <= 1000`

```cpp
class Solution {
public:
    int dfs(int i, int j, vector<vector<int>>& land, int r, int c, vector<vector<int>>& vis){
        // cout<<"i: "<<i<<" j: "<<j<<endl;
        if(land[i][j] == 0 && vis[i][j] == 0){
            vis[i][j] = 1;
            int cnt = 0;
            for(int ii=-1; ii<=1; ii++){
                for(int jj=-1; jj<=1; jj++){
                    if(i+ii >= 0 && i+ii < r && j+jj >= 0 && j+jj < c){
                        cnt += dfs(i+ii, j+jj, land, r, c, vis);
                    }
                }
            }
            return 1 + cnt;
        }
        return 0;
    }
    vector<int> pondSizes(vector<vector<int>>& land) {
        vector<int> ans;
        int r = land.size();
        if(r == 0) return ans;
        int c = land[0].size();
        vector<vector<int>> vis(r, vector<int>(c, 0));
        int cnt = 0;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(land[i][j] == 0 && vis[i][j] == 0){
                    cnt++;
                    // cout<<"cnt: "<<cnt<<endl;
                    ans.push_back(dfs(i, j, land, r, c, vis));
                }
            }
        }
        sort(ans.begin(), ans.end());
        return ans;
    }
};
```

### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

难度：中等

根据一棵树的中序遍历与后序遍历构造二叉树。

**注意:**
你可以假设树中没有重复的元素。

例如，给出
```
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]
```
返回如下的二叉树：
```
    3
   / \
  9  20
    /  \
   15   7
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        int inlen = inorder.size();
        if(inlen == 0) return nullptr;
        int postlen = postorder.size();
        int rootval = postorder[postlen-1];
        TreeNode* root = new TreeNode(rootval);
        vector<int> leftinorder;
        vector<int> leftpostorder;
        int i=0;
        for(; i<inlen; i++){
            if(inorder[i] == rootval)
                break;
            else leftinorder.push_back(inorder[i]);
        }
        for(int ii=0; ii<i; ii++){
            leftpostorder.push_back(postorder[ii]);
        }
        vector<int> rightinorder;
        vector<int> rightpostorder;
        for(int ii=i; ii<postlen-1; ii++){
            rightpostorder.push_back(postorder[ii]);
        }
        for(i++; i<inlen; i++){
            rightinorder.push_back(inorder[i]);
        }
        root->left = buildTree(leftinorder, leftpostorder);
        root->right = buildTree(rightinorder, rightpostorder);
        return root;
    }
};
```

### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

难度：中等 # 远古

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**示例 1：**
```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```
**示例 2：**
```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```
**提示：**

+ `m == grid.length`
+ `n == grid[i].length`
+ `1 <= m, n <= 300`
+ `grid[i][j] 的值为 '0' 或 '1'`

```cpp
class Solution {
public:
    vector<int> di = {-1, 0, 1, 0};
    vector<int> dj = {0, -1, 0, 1};
    void dfs(vector<vector<char>>& grid, vector<vector<int> >& v, int& m, int& n, int i, int j){
        v[i][j] = 1;
        if(grid[i][j] == '1'){
            for(int d=0; d<4; d++)
                if(i+di[d] >= 0 && i+di[d] < m && j+dj[d] >= 0 && j+dj[d] < n && v[i+di[d]][j+dj[d]] == 0 && grid[i+di[d]][j+dj[d]] == '1'){
                    dfs(grid, v, m, n, i+di[d], j+dj[d]);
                }
        }
    }
    int numIslands(vector<vector<char>>& grid) {
        int m = grid.size();
        if(m == 0)
            return 0;
        int n = grid[0].size();
        if(n == 0)
            return 0;
        vector<vector<int> > v(m, vector<int> (n, 0));
        int cnt = 0;
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(!v[i][j]){
                    if(grid[i][j] == '1')
                        cnt++;
                    dfs(grid, v, m, n, i, j);
                }
                    
            }
        }
        return cnt;
    }
};
```
### [1254. 统计封闭岛屿的数目](https://leetcode-cn.com/problems/number-of-closed-islands/)

难度：中等 # 2020.11.17

有一个二维矩阵 `grid` ，每个位置要么是陆地（记号为 `0` ）要么是水域（记号为 `1` ）。

我们从一块陆地出发，每次可以往上下左右 4 个方向相邻区域走，能走到的所有陆地区域，我们将其称为一座「**岛屿**」。

如果一座岛屿 **完全** 由水域包围，即陆地边缘上下左右所有相邻区域都是水域，那么我们将其称为 「**封闭岛屿**」。

请返回封闭岛屿的数目。

**示例 1：**

```
输入：grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
输出：2
解释：
灰色区域的岛屿是封闭岛屿，因为这座岛屿完全被水域包围（即被 1 区域包围）。
```
**示例 2：**

```
输入：grid = [[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]
输出：1
```
**示例 3：**

```
输入：grid = [[1,1,1,1,1,1,1],
             [1,0,0,0,0,0,1],
             [1,0,1,1,1,0,1],
             [1,0,1,0,1,0,1],
             [1,0,1,1,1,0,1],
             [1,0,0,0,0,0,1],
             [1,1,1,1,1,1,1]]
输出：2
```
**提示：**

+ `1 <= grid.length, grid[0].length <= 100`
+ `0 <= grid[i][j] <=1`

```cpp
class Solution {
public:
    bool isedge(int i, int j, int& r, int& c){
        if(i==0 || i==r-1 || j==0 || j==c-1)
            return true;
        return false;
    }
    bool dfs(int i, int j, vector<vector<int>>& grid, int r, int c, vector<vector<int>>& vis){ // 如果遍历到的节点是边缘则返回false
        int dx[4] = {0, -1, 0, 1};
        int dy[4] = {1, 0, -1, 0};
        vis[i][j] = 1;
        bool flag = !isedge(i, j, r, c); // 记录是否遍历路径中有位于边缘节点
        for(int k=0; k<4; k++){
            int ii = i+dx[k];
            int jj = j+dy[k];
            if(ii >= 0 && ii < r && jj >= 0 && jj < c && grid[ii][jj] == 0 && vis[ii][jj] == 0){
                flag = dfs(ii, jj, grid, r, c, vis) && flag;
            }
        }
        return flag;
    }
    int closedIsland(vector<vector<int>>& grid) {
        int r = grid.size();
        if(r == 0) return 0;
        int c = grid[0].size();
        vector<vector<int>> vis(r, vector<int>(c, 0));
        int ans = 0;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(grid[i][j] == 0 && vis[i][j] == 0){
                    ans++;
                    if(!dfs(i, j, grid, r, c, vis))
                        ans--; // 路径中有边缘，再减回去
                }
            }
        }
        return ans;
    }
};
```

### [827. 最大人工岛](https://leetcode-cn.com/problems/making-a-large-island/)

难度：困难 # 2020.11.22

在二维地图上，`0`代表海洋，`1`代表陆地，我们最多只能将一格`0`海洋变成`1`变成陆地。

进行填海之后，地图上最大的岛屿面积是多少？（上、下、左、右四个方向相连的`1`可形成岛屿）

**示例 1:**

```
输入: [[1, 0], [0, 1]]
输出: 3
解释: 将一格0变成1，最终连通两个小岛得到面积为 3 的岛屿。
```
**示例 2:**

```
输入: [[1, 1], [1, 0]]
输出: 4
解释: 将一格0变成1，岛屿的面积扩大为 4。
```
**示例 3:**

```
输入: [[1, 1], [1, 1]]
输出: 4
解释: 没有0可以让我们变成1，面积依然为 4。
```
**说明:**

+ `1 <= grid.length = grid[0].length <= 50`
+ `0 <= grid[i][j] <= 1`

```cpp
class Solution {
public:
    int dfs(int i, int j, vector<vector<int>>& grid, int r, int c, vector<vector<int>>& vis, vector<vector<int>>& idx, int cnt){
        // cout<<"i: "<<i<<" j: "<<j<<endl;
        vis[i][j] = 1;
        idx[i][j] = cnt;
        // cout<<"cnt: "<<cnt<<endl; 
        int dx[4] = {0, -1, 0, 1};
        int dy[4] = {-1, 0, 1, 0};
        int area = 1; // 返回当前连通块大小
        for(int k=0; k<4; k++){
            int ii = i+dx[k], jj = j+dy[k];
            if(ii>=0 && ii<r && jj>=0 && jj<c && grid[ii][jj] == 1 && vis[ii][jj] == 0){
                area += dfs(ii, jj, grid, r, c, vis, idx, cnt);
            }
        }
        return area;
    }
    int largestIsland(vector<vector<int>>& grid) {
        int r = grid.size();
        int c = grid[0].size();
        vector<vector<int>> vis(r, vector<int>(c, 0));
        vector<vector<int>> idx(r, vector<int>(c, 0)); // 每个位置对应连通块索引
        vector<int> idx_area; // 每个连通块索引对应的连通块大小
        int maxarea = 0;
        int cnt = 0;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(grid[i][j] == 1 && vis[i][j] == 0){
                    cnt++; // 索引从1开始计
                    int area = dfs(i, j, grid, r, c, vis, idx, cnt);
                    // cout<<"area: "<<area<<endl;
                    idx_area.push_back(area);
                    maxarea = max(maxarea, area);
                }
            }
        }
        if(maxarea == 0) return 1; // 都是0的情况
        // cout<<"maxarea: "<<maxarea<<endl;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(grid[i][j] == 0){
                    // cout<<"i: "<<i<<" j: "<<j<<endl;
                    int area = 1;
                    set<int> s;
                    int dx[4] = {0, 1, 0, -1};
                    int dy[4] = {-1, 0, 1, 0};
                    for(int k=0; k<4; k++){
                        int ii = i+dx[k], jj = j+dy[k];
                        if(ii>=0 && ii<r && jj>=0 && jj<c && idx[ii][jj]>0)
                            s.insert(idx[ii][jj]);
                    }
                    for(set<int>::iterator it = s.begin(); it != s.end(); it++){
                        // cout<<*it<<"'s area is: "<<idx_area[*it-1]<<endl;
                        area += idx_area[*it-1];
                    }
                    maxarea = max(maxarea, area);
                }
            }
        }
        return maxarea;
    }
};
```

### [1319. 连通网络的操作次数](https://leetcode-cn.com/problems/number-of-operations-to-make-network-connected/)

难度：中等 # 2021.01.23

用以太网线缆将 `n` 台计算机连接成一个网络，计算机的编号从 `0` 到 `n-1`。线缆用 `connections` 表示，其中 `connections[i] = [a, b]` 连接了计算机 `a` 和 `b`。

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线 `connections`，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 -1 。 

示例 1：

![sample_1_1677](./images/sample_1_1677.png)
```
输入：n = 4, connections = [[0,1],[0,2],[1,2]]
输出：1
解释：拔下计算机 1 和 2 之间的线缆，并将它插到计算机 1 和 3 上。
```
**示例 2：**

![sample_2_1677](./images/sample_2_1677.png)

```
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2],[1,3]]
输出：2
```

**示例 3：**

```
输入：n = 6, connections = [[0,1],[0,2],[0,3],[1,2]]
输出：-1
解释：线缆数量不足。
```

**示例 4：**

```
输入：n = 5, connections = [[0,1],[0,2],[3,4],[2,3]]
输出：0
```

提示：

+ `1 <= n <= 10^5`
+ `1 <= connections.length <= min(n*(n-1)/2, 10^5)`
+ `connections[i].length == 2`
+ `0 <= connections[i][0], connections[i][1] < n`
+ `connections[i][0] != connections[i][1]`
+ 没有重复的连接。
+ 两台计算机不会通过多条线缆连接。

```cpp
class Solution {
public:
    void dfs(int x, vector<int>& v, vector<vector<int>>& adj){
        v[x] = 1;
        int len = adj[x].size();
        for(int i=0; i<len; i++){
            if(!v[adj[x][i]]) dfs(adj[x][i], v, adj);
        }
    }
    int makeConnected(int n, vector<vector<int>>& connections) {
        int len = connections.size();
        if(len < n-1) return -1;
        vector<vector<int>> adj(n, vector<int>());
        for(int i=0; i<len; i++){
            adj[connections[i][0]].push_back(connections[i][1]);
            adj[connections[i][1]].push_back(connections[i][0]);
        }
        vector<int> v(n, 0);
        int cnt = 0;
        for(int i=0; i<n; i++){
            if(!v[i]){
                cnt++;
                dfs(i, v, adj);
            }
        }
        return cnt-1;
    }
};
```

### [395. 至少有 K 个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

难度：中等 # 2021.03.05

给你一个字符串 `s` 和一个整数 `k` ，请你找出 `s` 中的最长子串， 要求该子串中的每一字符出现次数都不少于 `k` 。返回这一子串的长度。

**示例 1：**
```
输入：s = "aaabb", k = 3
输出：3
解释：最长子串为 "aaa" ，其中 'a' 重复了 3 次。
```
**示例 2：**
```
输入：s = "ababbc", k = 2
输出：5
解释：最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
```

**提示：**

+ `1 <= s.length <= 10^4`
+ `s 仅由小写英文字母组成`
+ `1 <= k <= 10^5`

```cpp
class Solution {
public:
    int dfs(const string& s, int l, int r, int k) {
        unordered_map<char, int> cnt;
        for(int i=l; i<=r; i++){
            cnt[s[i]]++;
        }
        int ans = 0;
        int i = l;
        while(i<=r){
            while(i<=r && cnt[s[i]]>0 && cnt[s[i]]<k){
                i++;
            }
            if(i>r) break;
            int first = i;
            while(i<=r && cnt[s[i]]>=k){
                i++;
            }
            int second = i - 1;
            if(first == l && second == r) return r-l+1;
            int x = dfs(s, first, second, k);
            ans = max(ans, x);
        }
        return ans;
    }
    int longestSubstring(string s, int k) {
        int len = s.length();
        return dfs(s, 0, len-1, k);
    }
};
```

### [341. 扁平化嵌套列表迭代器](https://leetcode-cn.com/problems/flatten-nested-list-iterator/)

难度：中等 # 2021.03.23

给你一个嵌套的整型列表。请你设计一个迭代器，使其能够遍历这个整型列表中的所有整数。

列表中的每一项或者为一个整数，或者是另一个列表。其中列表的元素也可能是整数或是其他列表。

**示例 1：**
```
输入: [[1,1],2,[1,1]]
输出: [1,1,2,1,1]
解释: 通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,1,2,1,1]。
```
**示例 2：**
```
输入: [1,[4,[6]]]
输出: [1,4,6]
解释: 通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,4,6]。
```

```cpp
/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * class NestedInteger {
 *   public:
 *     // Return true if this NestedInteger holds a single integer, rather than a nested list.
 *     bool isInteger() const;
 *
 *     // Return the single integer that this NestedInteger holds, if it holds a single integer
 *     // The result is undefined if this NestedInteger holds a nested list
 *     int getInteger() const;
 *
 *     // Return the nested list that this NestedInteger holds, if it holds a nested list
 *     // The result is undefined if this NestedInteger holds a single integer
 *     const vector<NestedInteger> &getList() const;
 * };
 */

class NestedIterator {
private:
    vector<int> vals;
    vector<int>::iterator cur;

    void dfs(const vector<NestedInteger> &nestedList) {
        for (auto &nest : nestedList) {
            if (nest.isInteger()) {
                vals.push_back(nest.getInteger());
            } else {
                dfs(nest.getList());
            }
        }
    }

public:
    NestedIterator(vector<NestedInteger> &nestedList) {
        dfs(nestedList);
        cur = vals.begin();
    }

    int next() {
        return *cur++;
    }

    bool hasNext() {
        return cur != vals.end();
    }
};

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i(nestedList);
 * while (i.hasNext()) cout << i.next();
 */
```

### [1011. 在 D 天内送达包裹的能力](https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/)

难度：中等 # 2021.04.26

传送带上的包裹必须在 `D` 天内从一个港口运送到另一个港口。

传送带上的第 `i` 个包裹的重量为 `weights[i]`。每一天，我们都会按给出重量的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。

返回能在 `D` 天内将传送带上的所有包裹送达的船的最低运载能力。

**示例 1：**
```
输入：weights = [1,2,3,4,5,6,7,8,9,10], D = 5
输出：15
解释：
船舶最低载重 15 就能够在 5 天内送达所有包裹，如下所示：
第 1 天：1, 2, 3, 4, 5
第 2 天：6, 7
第 3 天：8
第 4 天：9
第 5 天：10

请注意，货物必须按照给定的顺序装运，因此使用载重能力为 14 的船舶并将包装分成 (2, 3, 4, 5), (1, 6, 7), (8), (9), (10) 是不允许的。 
```
**示例 2：**
```
输入：weights = [3,2,2,4,1,4], D = 3
输出：6
解释：
船舶最低载重 6 就能够在 3 天内送达所有包裹，如下所示：
第 1 天：3, 2
第 2 天：2, 4
第 3 天：1, 4
```
**示例 3：**
```
输入：weights = [1,2,3,1,1], D = 4
输出：3
解释：
第 1 天：1
第 2 天：2
第 3 天：3
第 4 天：1, 1
```

**提示：**

1. `1 <= D <= weights.length <= 50000`
2. `1 <= weights[i] <= 500`

```cpp
class Solution {
public:
    int shipWithinDays(vector<int>& weights, int D) { // 二分查找
        int left = *max_element(weights.begin(), weights.end()), right = accumulate(weights.begin(), weights.end(), 0);
        while(left < right){
            int mid = (left + right) / 2;
            // need 为需要运送的天数
            // cur 为当前这一天已经运送的包裹重量之和
            int need = 1, cur = 0;
            for(int weight: weights){
                if(cur + weight > mid){
                    need++;
                    cur = 0;
                }
                cur += weight;
            }
            if(need <= D){
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
};
```

### [1723. 完成所有工作的最短时间](https://leetcode-cn.com/problems/find-minimum-time-to-finish-all-jobs/)

难度：困难 #2021.05.08

给你一个整数数组 `jobs` ，其中 `jobs[i]` 是完成第 `i` 项工作要花费的时间。

请你将这些工作分配给 `k` 位工人。所有工作都应该分配给工人，且每项工作只能分配给一位工人。工人的 **工作时间** 是完成分配给他们的所有工作花费时间的总和。请你设计一套最佳的工作分配方案，使工人的 **最大工作时间** 得以 **最小化** 。

返回分配方案中尽可能 **最小** 的 **最大工作时间** 。

**示例 1：**
```
输入：jobs = [3,2,3], k = 3
输出：3
解释：给每位工人分配一项工作，最大工作时间是 3 。
```
**示例 2：**
```
输入：jobs = [1,2,4,7,8], k = 2
输出：11
解释：按下述方式分配工作：
1 号工人：1、2、8（工作时间 = 1 + 2 + 8 = 11）
2 号工人：4、7（工作时间 = 4 + 7 = 11）
最大工作时间是 11 。
```

**提示：**

+ `1 <= k <= jobs.length <= 12`
+ `1 <= jobs[i] <= 10^7`

```cpp
class Solution {
public:
    void dfs(vector<int>& jobs, vector<int>& worktime, int start, int len, int k, int& ans){
        if(start == len){
            ans = min(ans, *max_element(worktime.begin(), worktime.end()));
        }
        else{
            for(int i=0; i<k; i++){
                if(worktime[i] + jobs[start]> ans) continue;
                worktime[i] += jobs[start];
                dfs(jobs, worktime, start+1, len, k, ans);
                worktime[i] -= jobs[start];
                if(worktime[i] == 0) break; // 每个人都一样，一个人到头就够了
            }
        }
    }
    int minimumTimeRequired(vector<int>& jobs, int k) {
        vector<int> worktime(k, 0);
        int len = jobs.size();
        int ans = INT_MAX;
        dfs(jobs, worktime, 0, len, k, ans);
        return ans;
    }
};
```



## 链表

### [面试题 02.02. 返回倒数第 k 个节点](https://leetcode-cn.com/problems/kth-node-from-end-of-list-lcci/)

难度：简单 # 2020.11.16

实现一种算法，找出单向链表中倒数第 k 个节点。返回该节点的值。

**注意：**本题相对原题稍作改动

示例：
```
输入： 1->2->3->4->5 和 k = 2
输出： 4
```
说明：

给定的 k 保证是有效的。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    int kthToLast(ListNode* head, int k) {
        vector<int> a;
        for(ListNode* p=head; p; p=p->next){
            a.push_back(p->val);
        }
        return a[a.size()-k];
    }
};
```

### [面试题 02.06. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list-lcci/)

难度：简单 # 远古

编写一个函数，检查输入的链表是否是回文的。

**示例 1：**

```
输入： 1->2
输出： false 
```

**示例 2：**

```
输入： 1->2->2->1
输出： true
```

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        vector<int> v;
        for(ListNode *p=head; p!=nullptr; p = p->next){
            v.push_back(p->val);
        }
        int len = v.size();
        for(int i=0; i<len/2; i++){
            if(v[i] != v[len-1-i]) return false;
        }
        return true;
    }
};
```

### [面试题 02.04. 分割链表](https://leetcode-cn.com/problems/partition-list-lcci/)

难度：中等 # 2020.11.18

编写程序以 x 为基准分割链表，使得所有小于 x 的节点排在大于或等于 x 的节点之前。如果链表中包含 x，x 只需出现在小于 x 的元素之后(如下所示)。分割元素 x 只需处于“右半部分”即可，其不需要被置于左右两部分之间。

**示例:**

```
输入: head = 3->5->8->5->10->2->1, x = 5
输出: 3->1->2->10->5->5->8
```

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        if(!head || !head->next) return head;
        ListNode* p=head;
        ListNode* bh=new ListNode(0);
        ListNode* sh=new ListNode(0);
        ListNode* bp=bh;
        ListNode* sp=sh;
        while(p){
            if(p->val>=x){
                bp->next=p;
                bp=bp->next;
                p=p->next;
                bp->next=nullptr;
            }else {
                sp->next=p;
                sp=sp->next;
                p=p->next;
                sp->next=nullptr;
            }  
        }
        sp->next=bh->next;
        return sh->next;
    }
};
```

### [328. 奇偶链表](https://leetcode-cn.com/problems/odd-even-linked-list/)

难度：中等 # 2020.11.22

给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

**示例 1:**
```
输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL
```
**示例 2:**

```
输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL
```
**说明:**

+ `应当保持奇数节点和偶数节点的相对顺序。`
+ `链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推。`

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        if (head == nullptr) {
            return head;
        }
        ListNode* evenHead = head->next;
        ListNode* odd = head;
        ListNode* even = evenHead;
        while (even != nullptr && even->next != nullptr) {
            odd->next = even->next;
            odd = odd->next;
            even->next = odd->next;
            even = even->next;
        }
        odd->next = evenHead;
        return head;
    }
};
```

### [面试题 02.08. 环路检测](https://leetcode-cn.com/problems/linked-list-cycle-lcci/)

难度：中等 # 2020.11.22

给定一个链表，如果它是有环链表，实现一个算法返回环路的开头节点。

如果链表中有某个节点，可以通过连续跟踪`next`指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数`pos`来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果`pos`是`-1`，则在该链表中没有环。**注意：pos 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**示例 1：**

![circularlinkedlist](./images/circularlinkedlist.png)
```
输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。
```
**示例 2：**

![circularlinkedlist_test2](./images/circularlinkedlist_test2.png)
```
输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。
```
**示例 3：**

![circularlinkedlist_test3](./images/circularlinkedlist_test3.png)
```
输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。
```

**进阶：**

+ 你是否可以不用额外空间解决此题？

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) { // 起点到入口距离为a，入口到相遇点距离为b，相遇点到入口距离为c
        ListNode *slow = head, *fast = head; // 2(a+b) = a+n(b+c)+b -> a+b = n(b+c) 
        while(fast != nullptr && fast->next != nullptr){ // 快慢指针，快指针每次走两步，慢指针每次走一步
            fast = fast->next->next;
            slow = slow->next;
            if(slow == fast){ // 先判断是否有环，确定有环之后才能找环的入口
                while(head != slow){ // 一个从起点开始，一个从相遇点开始每次走一步，直到再次相遇为止
                    head = head->next;
                    slow = slow->next;
                }
                return slow;
            }
        }
        return nullptr;
    }
};
```

### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

难度：困难 # 2020.11.23

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

**示例：**

给你这个链表：`1->2->3->4->5`

当 k = 2 时，应当返回: `2->1->4->3->5`

当 k = 3 时，应当返回: `3->2->1->4->5`

**说明：**

+ 你的算法只能使用常数的额外空间。
+ **你不能只是单纯的改变节点内部的值**，而是需要实际进行节点交换。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* h){ // 终点为nullptr的链表逆序
        // cout<<"reverse from: "<<h->val<<endl;
        ListNode *pf, *pb, *tmp;
        pf = h;
        if(!h || !h->next) return h;
        else{
            pb = pf->next;
            h->next = nullptr;
            while(pb){
                tmp = pb;
                pb = pb->next;
                tmp->next = pf;            
                pf = tmp;
            }
            h = pf;
            return h;
        }   
    }
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode *h = head; // 每k个的头结点
        ListNode *p = head; // 依次往下遍历
        ListNode *fakehead = new ListNode(0);
        ListNode *pp = fakehead; // 记录每k个之前的结点
        int cnt = 0;
        while(p){
            // cout<<"p: "<<p->val<<endl;
            cnt++;
            if(cnt % k == 0){ // 每k个局部翻转
                ListNode *tmp = p->next;
                p->next = nullptr;
                h = reverseList(h); // 返回逆序后的头结点
                pp->next = h; // 接上逆序后的这一段
                h = tmp; // h成为下一k个的起点
                while(pp->next){
                    pp = pp->next; // 此时pp成为下一个k个的前一个结点
                }
                p = tmp;
            }
            else p = p->next;
        }
        pp->next = h;
        return fakehead->next;
    }
};
```

### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

难度：困难 # 2020.11.24

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**示例 1：**
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```
**示例 2：**
```
输入：lists = []
输出：[]
```
**示例 3：**
```
输入：lists = [[]]
输出：[]
```
**提示：**

+ `k == lists.length`
+ `0 <= k <= 10^4`
+ `0 <= lists[i].length <= 500`
+ `-10^4 <= lists[i][j] <= 10^4`
+ `lists[i] 按升序排列`
+ `lists[i].length 的总和不超过 10^4`

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int len = lists.size();
        ListNode* fakehead = new ListNode(0);
        ListNode* p = fakehead;
        bool allend = false;
        while(!allend){
            allend = true; // 记录这一轮遍历是否都到链表终点
            int min_val = INT_MAX, min_idx = INT_MAX;
            for(int i=0; i<len; i++){
                if(lists[i]){
                    allend &= false;
                    if(lists[i]->val < min_val){
                        min_val = lists[i]->val;
                        min_idx = i;
                    }
                }
            }
            if(allend) break;
            p->next = new ListNode(min_val);
            p = p->next;
            lists[min_idx] = lists[min_idx]->next;
        }
        return fakehead->next;
    }
};
```

### [86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

难度：中等 # 2021.01.03

给你一个链表和一个特定值 `x` ，请你对链表进行分隔，使得所有小于 `x` 的节点都出现在大于或等于 `x` 的节点之前。

你应当保留两个分区中每个节点的初始相对位置。

 **示例：**
```
输入：head = 1->4->3->2->5->2, x = 3
输出：1->2->2->4->3->5
```
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* small = new ListNode(0);
        ListNode* smallHead = small;
        ListNode* large = new ListNode(0);
        ListNode* largeHead = large;
        while(head != nullptr){
            if(head->val < x){
                small->next = head;
                small = small->next;
            } else {
                large->next = head;
                large = large->next;
            }
            head = head->next;
        }
        large->next = nullptr;
        small->next = largeHead->next;
        return smallHead->next;
    }
};
```

### [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

难度：中等 # 2021.03.18

给你单链表的头指针 `head` 和两个整数 `left` 和 `right`，其中 `left <= right`。请你反转从位置 `left` 到位置 `right` 的链表节点，返回 **反转后的链表**。

**示例1：**
![rev2ex2](./images/rev2ex2.jpg)
```
输入：head = [1,2,3,4,5], left = 2, right = 4
输出：[1,4,3,2,5]
```
**示例 2：**
```
输入：head = [5], left = 1, right = 1
输出：[5]
```

**提示：**

+ 链表中节点数目为 `n`
+ `1 <= n <= 500`
+ `-500 <= Node.val <= 500`
+ `1 <= left <= right <= n`

**进阶：** 你可以使用一趟扫描完成反转吗？

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    void reverseLinkedList(ListNode *head){
        ListNode *pre = nullptr;
        ListNode *p = head;
        while(p != nullptr){
            ListNode *tmp = p->next;
            p->next = pre;
            pre = p;
            p = tmp;
        }
    }
    ListNode *reverseBetween(ListNode *head, int left, int right) {
        ListNode *head_node = new ListNode(0);
        head_node->next = head;

        ListNode *pre = head_node;
        for(int i=0; i<left-1; i++) pre = pre->next; // left前一个节点
        ListNode *rightNode = pre;
        for(int i=0; i<right-left+1; i++) rightNode = rightNode->next;

        ListNode *leftNode = pre->next;
        ListNode *post = rightNode->next; // right后一个节点

        pre->next = nullptr;
        rightNode->next = nullptr;

        reverseLinkedList(leftNode);
        
        pre->next = rightNode;
        leftNode->next = post;
        return head_node->next;
    }
};
```

### [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

难度：中等 # 2021.03.25

存在一个按升序排列的链表，给你这个链表的头节点 `head` ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 **没有重复出现** 的数字。

返回同样按升序排列的结果链表。

**示例 1：**
![linkedlist1](./images/linkedlist1.jpg)
```
输入：head = [1,2,3,3,4,4,5]
输出：[1,2,5]
```
**示例 2：**
![linkedlist2](./images/linkedlist2.jpg)

```
输入：head = [1,1,1,2,3]
输出：[2,3]
```

**提示：**

+ 链表中节点数目在范围 `[0, 300]` 内
+ `-100 <= Node.val <= 100`
+ 题目数据保证链表已经按升序排列

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head) return head;        
        ListNode* head_node = new ListNode(0, head);
        ListNode* cur = head_node;
        while(cur->next && cur->next->next){
            if(cur->next->val == cur->next->next->val){
                int x = cur->next->val;
                while(cur->next && cur->next->val == x){
                    cur->next = cur->next->next;
                }
            } else {
                cur = cur->next;
            }
        }
        return head_node->next;
    }
};
```

### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

难度：简单 # 2021.03.26

存在一个按升序排列的链表，给你这个链表的头节点 `head` ，请你删除所有重复的元素，使每个元素 **只出现一次** 。

返回同样按升序排列的结果链表。

**示例 1：**
![list1](./images/list1.jpg)

```
输入：head = [1,1,2]
输出：[1,2]
```
**示例 2：**
![list2](./images/list2.jpg)

```
输入：head = [1,1,2,3,3]
输出：[1,2,3]
```
**提示：**

+ 链表中节点数目在范围 `[0, 300]` 内
+ `-100 <= Node.val <= 100`
+ 题目数据保证链表已经按升序排列

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode* p = head;
        while(p && p->next){
            if(p->next->val == p->val){
                p->next = p->next->next;
            } else {
                p = p->next;
            }
        }
        return head;
    }
};
```

### [61. 旋转链表](https://leetcode-cn.com/problems/rotate-list/)

难度：中等 # 2021.03.27

给你一个链表的头节点 `head` ，旋转链表，将链表每个节点向右移动 `k` 个位置。

**示例 1：**
![rotate1](./images/rotate1.jpg)
```
输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]
```
**示例 2：**
![roate2](./images/roate2.jpg)
```
输入：head = [0,1,2], k = 4
输出：[2,0,1]
```

**提示：**

+ 链表中节点的数目在范围 `[0, 500]` 内
+ `-100 <= Node.val <= 100`
+ `0 <= k <= 2 * 10^9`

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if(!head) return head;
        
        ListNode* p = head;
        int cnt = 1;
        while(p->next != nullptr){
            cnt++;
            p = p->next;
        }
        p->next = head;

        int aim = cnt - (k % cnt);
        int i = 1;
        ListNode* p2 = head;
        while(i < aim){
            p2 = p2->next;
            i++;
        }
        ListNode* ans = p2->next;
        p2->next = nullptr;
        return ans;
    }
};
```



## 广度优先搜索

### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

难度：中等 # 2020.11.24

给你一个二叉树，请你返回其按 **层序遍历** 得到的节点值。（即逐层地，从左到右访问所有节点）。

**示例：**
二叉树：`[3,9,20,null,null,15,7]`,
```
    3
   / \
  9  20
    /  \
   15   7
```
返回其层次遍历结果：
```
[
  [3],
  [9,20],
  [15,7]
]
```

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    void bfs(queue<TreeNode*> &q, queue<int> &q_level, vector<vector<int>> &ans){
        TreeNode* root = q.front();
        q.pop();
        int level = q_level.front();
        q_level.pop();
        vector<int> v;
        if(ans.size() <= level) ans.push_back(v);
        ans[level].push_back(root->val);
        if(root->left){
            q.push(root->left);
            q_level.push(level+1);
        }
        if(root->right){
            q.push(root->right);
            q_level.push(level+1);
        }
    }
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q;
        queue<int> q_level; // 同时记录层数
        vector<vector<int>> ans;
        if(!root) return ans;
        q.push(root);
        q_level.push(0);
        while(!q.empty()){
            bfs(q, q_level, ans);
        }
        return ans;
    }
};
```

### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

难度：简单 # 2020.11.26

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**说明：**叶子节点是指没有子节点的节点。

**示例 1：**

![ex_depth](./images/ex_depth.jpg)
```
输入：root = [3,9,20,null,null,15,7]
输出：2
```
**示例 2：**
```
输入：root = [2,null,3,null,4,null,5,null,6]
输出：5
```

**提示：**

+ 树中节点数的范围在 `[0, 105]` 内
+ `-1000 <= Node.val <= 1000`

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool bfs(queue<TreeNode*> &q, queue<int> &q_depth, int &depth){ // 返回是否该节点的子节点有nullptr
        TreeNode* root = q.front();
        q.pop();
        // cout<<root->val<<endl;
        depth = q_depth.front();
        q_depth.pop();
        if(!root->left && !root->right) return true;
        if(root->left){
            q.push(root->left);
            q_depth.push(depth+1);
        }
        if(root->right){
            q.push(root->right);
            q_depth.push(depth+1);
        }
        return false;
    }
    int minDepth(TreeNode* root) {
        queue<TreeNode*> q;
        queue<int> q_depth; // 同时记录深度
        int depth = 0;
        if(!root) return depth;
        q.push(root);
        q_depth.push(1);
        depth = 1;
        while(!q.empty()){
            if(bfs(q, q_depth, depth)) break;
        }
        return depth;
    }
};
```

### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

难度：中等 # 2020.11.26

给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：

1. 每次转换只能改变一个字母。
2. 转换过程中的中间单词必须是字典中的单词。

**说明:**

+ 如果不存在这样的转换序列，返回 0。
+ 所有单词具有相同的长度。
+ 所有单词只由小写字母组成。
+ 字典中不存在重复的单词。
+ 你可以假设 *beginWord* 和 *endWord* 是非空的，且二者不相同。

**示例 1:**

```
输入:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
输出: 5
解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     返回它的长度 5。
```
**示例 2:**

```
输入:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]
输出: 0
解释: endWord "cog" 不在字典中，所以无法进行转换。
```

```cpp
class Solution {
public:
    //确定俩个单词能否互相转换
    bool exchange_word(const string& a, const string& b){
        int cnt = 0;
        int len = a.size();
        for(int i=0; i<len; i++){
            if(a[i] != b[i]) cnt++;
            if(cnt > 1) return false;
        }
        return cnt == 1;
    }
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> word_set(wordList.begin(), wordList.end()); // vector转set
        if(word_set.count(endWord) == 0) return 0;         
        queue<string> q;
        q.push(beginWord);
        word_set.erase(beginWord);
        int cnt = 1;
        while(!q.empty()){
            int size = q.size(); // 当前队列的长度即为每层节点数量
            for(int j = 0; j < size; j++){ // 循环该层所有节点
                string str = q.front(); q.pop();  
                // vector<string> tmp; // 保存str改变一个字母能够转换的单词，后面一起从集合中删除
                // for(string word : word_set){ // C++11 set遍历
                //     if(exchange_word(str, word)){
                //         if(word == endWord) return cnt + 1;
                //         tmp.push_back(word);
                //         q.push(word);
                //     }
                // }
                // for (string str : tmp) word_set.erase(str); // 将访问的tmp中元素删除
                for(unordered_set<string>::iterator it=word_set.begin(); it!=word_set.end();){ // set边遍历边删除的方式
                    if(exchange_word(str, *it)){
                        if(*it == endWord) return cnt + 1;
                        q.push(*it);
                        // it = word_set.erase(it);
                        word_set.erase(it++);
                    }
                    else it++;
                }
            }
            cnt++;
        }
        return 0;
    }
};
```

### [310. 最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)

难度：中等 # 2020.12.03

树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。

给你一棵包含 `n` 个节点的数，标记为 `0` 到 `n - 1` 。给定数字 `n` 和一个有 `n - 1` 条无向边的 `edges` 列表（每一个边都是一对标签），其中 `edges[i] = [ai, bi]` 表示树中节点 `ai` 和 `bi` 之间存在一条无向边。

可选择树中任何一个节点作为根。当选择节点 `x` 作为根节点时，设结果树的高度为 `h` 。在所有可能的树中，具有最小高度的树（即，`min(h)`）被称为 **最小高度树** 。

请你找到所有的 **最小高度树** 并按 **任意顺序** 返回它们的根节点标签列表。

树的 **高度** 是指根节点和叶子节点之间最长向下路径上边的数量。

**示例 1：**

![e1](./images/e1.jpg)
```
输入：n = 4, edges = [[1,0],[1,2],[1,3]]
输出：[1]
解释：如图所示，当根是标签为 1 的节点时，树的高度是 1 ，这是唯一的最小高度树。
```

**示例 2：**

![e2](./images/e2.jpg)
```
输入：n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
输出：[3,4]
```
**示例 3：**
```
输入：n = 1, edges = []
输出：[0]
```
**示例 4：**
```
输入：n = 2, edges = [[0,1]]
输出：[0,1]
```
**提示：**

+ `1 <= n <= 2 * 104`
+ `edges.length == n - 1`
+ `0 <= ai, bi < n`
+ `ai != bi`
+ 所有 `(ai, bi)` 互不相同
+ 给定的输入 **保证** 是一棵树，并且 **不会有重复的边**

```cpp
class Solution {
public:
    // 对图入度入度/出度为1的节点逐渐剪枝，最终可以将图化简为一条链，链长度为偶数则有两个解，链长度为奇数则有一个解。
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) { // 遍历每个节点作初始节点会超时
        if(n == 1)
            return {0};
        vector<vector<int>> adjacent_list(n);
        vector<int> degree(n);
        for(auto& edge: edges){ // auto 和 auto& 都行
            adjacent_list[edge[0]].push_back(edge[1]);
            adjacent_list[edge[1]].push_back(edge[0]);
            degree[edge[0]]++;
            degree[edge[1]]++;
        }
        queue<int>q;
        for(int i = 0; i < n; i++){
            if(degree[i] == 1){
                q.push(i);
            }
        }
        vector<int> ans;
        while(!q.empty()){
            ans.clear();
            while(!q.empty()){
                ans.push_back(q.front());
                q.pop();
            }
            for(int& u : ans){ // 对于每一个度数为1的节点
                for(int& v: adjacent_list[u]){ // 把与它相邻的节点度数--
                    degree[v]--;
                    if(degree[v] == 1){ // 新的节点度数为1就加入队列
                        q.push(v);
                    }
                }
            }
        }
        return ans;
    }
};
```

### [1306. 跳跃游戏 III](https://leetcode-cn.com/problems/jump-game-iii/)

难度：中等 # 2020.12.05

这里有一个非负整数数组 `arr`，你最开始位于该数组的起始下标 `start` 处。当你位于下标 `i` 处时，你可以跳到 `i + arr[i]` 或者 `i - arr[i]`。

请你判断自己是否能够跳到对应元素值为 0 的 **任一** 下标处。

注意，不管是什么情况下，你都无法跳到数组之外。


**示例 1：**
```
输入：arr = [4,2,3,0,3,1,2], start = 5
输出：true
解释：
到达值为 0 的下标 3 有以下可能方案： 
下标 5 -> 下标 4 -> 下标 1 -> 下标 3 
下标 5 -> 下标 6 -> 下标 4 -> 下标 1 -> 下标 3 
```
**示例 2：**
```
输入：arr = [4,2,3,0,3,1,2], start = 0
输出：true 
解释：
到达值为 0 的下标 3 有以下可能方案： 
下标 0 -> 下标 4 -> 下标 1 -> 下标 3
```
**示例 3：**
```
输入：arr = [3,0,2,1,2], start = 2
输出：false
解释：无法到达值为 0 的下标 1 处。 
```

**提示：**

+ `1 <= arr.length <= 5 * 10^4`
+ `0 <= arr[i] < arr.length`
+ `0 <= start < arr.length`

```cpp
class Solution {
public:
    bool canReach(vector<int>& arr, int start) {
        int len = arr.size();
        vector<int> vis(len, 0);
        queue<int> q;
        q.push(start);
        while(!q.empty()){
            int qlen = q.size();
            for(int i=0; i<qlen; i++){
                int idx = q.front(); q.pop();
                vis[idx] = 1;
                // cout<<idx<<endl;
                if(arr[idx] == 0) return true;
                int idx_left = idx - arr[idx];
                int idx_right = idx + arr[idx];
                if(idx_left >= 0 && vis[idx_left] == 0)
                    q.push(idx_left);
                if(idx_right < len && vis[idx_right] == 0)
                    q.push(idx_right);
            }
        }
        return false;
    }
};
```

### [407. 接雨水 II](https://leetcode-cn.com/problems/trapping-rain-water-ii/)

难度：困难 # 2020.12.05

给你一个 `m x n` 的矩阵，其中的值均为非负整数，代表二维高度图每个单元的高度，请计算图中形状最多能接多少体积的雨水。 

**示例：**
```
给出如下 3x6 的高度图:
[
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]
返回 4 。
```
![rainwater_empty](./images/rainwater_empty.png)

如上图所示，这是下雨前的高度图 `[[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]` 的状态。

![rainwater_fill](./images/rainwater_fill.png)

下雨后，雨水将会被存储在这些方块中。总的接雨水量是4。

**提示：**

+ `1 <= m, n <= 110`
+ `0 <= heightMap[i][j] <= 20000`

```cpp
struct Node {
    int i, j, h;
    Node(int x, int y, int z) : i(x), j(y), h(z) {}
    bool operator < (const Node &b) const {
        if (h > b.h) return true;
        return false;
    }
};

class Solution {
public:
    int trapRainWater(vector<vector<int>>& heightMap) {
        int r = heightMap.size();
        if(r == 0) return 0;
        int c = heightMap[0].size();
        priority_queue<Node> pq;
        vector<vector<int>> vis(r, vector<int>(c, 0));
        for(int i=0; i<r; i++){ // 将最外面一圈填入优先队列，向内收缩
            for(int j=0; j<c; j++){
                if(i==0 || i==r-1 || j==0 || j==c-1){
                    vis[i][j] = 1;
                    pq.push(Node(i, j, heightMap[i][j]));
                }
            }
        }
        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, 1, -1};
        int ans = 0;
        while(!pq.empty()){
            Node node = pq.top(); pq.pop();
            for(int i = 0; i < 4; i++){
                int x = node.i + dx[i], y = node.j + dy[i];
                if(x >= 0 && x < r && y >= 0 && y < c && !vis[x][y]){
                    vis[x][y] = 1;
                    pq.push(Node(x, y, max(node.h, heightMap[x][y]))); // 收缩边界的时候，填水则为外圈高度，不填水则为位置高度
                    ans += max(0, node.h - heightMap[x][y]); // 是否填水
                }
            }
        }
        return ans;
    }
};
```

### [913. 猫和老鼠](https://leetcode-cn.com/problems/cat-and-mouse/)

难度：困难 # 2020.12.09

两个玩家分别扮演猫（Cat）和老鼠（Mouse）在**无向**图上进行游戏，他们轮流行动。

该图按下述规则给出：`graph[a]` 是所有结点 `b` 的列表，使得 `ab` 是图的一条边。

老鼠从结点 1 开始并率先出发，猫从结点 2 开始且随后出发，在结点 0 处有一个洞。

在每个玩家的回合中，他们必须沿着与他们所在位置相吻合的图的一条边移动。例如，如果老鼠位于结点 `1`，那么它只能移动到 `graph[1]` 中的（任何）结点去。

此外，猫无法移动到洞（结点 0）里。

然后，游戏在出现以下三种情形之一时结束：

如果猫和老鼠占据相同的结点，猫获胜。
如果老鼠躲入洞里，老鼠获胜。
如果某一位置重复出现（即，玩家们的位置和移动顺序都与上一个回合相同），游戏平局。
给定 `graph`，并假设两个玩家都以最佳状态参与游戏，如果老鼠获胜，则返回 `1`；如果猫获胜，则返回 `2`；如果平局，则返回 `0`。

**示例：**
```
输入：[[2,5],[3],[0,4,5],[1,4,5],[2,3],[0,2,3]]
输出：0
解释：
4---3---1
|   |
2---5
 \ /
  0
```

**提示：**

1. `3 <= graph.length <= 200`
2. 保证 `graph[1]` 非空。
3. 保证 `graph[2]` 包含非零元素。

```cpp
class Solution {
public:
    int helper(vector<vector<int>>& graph, int t, int x, int y, vector<vector<vector<int>>>& dp) {  // 动态规划，t代表步数，x代表鼠位置，y代表猫位置
    	if (t == graph.size() * 2) return 0; // 博弈，走遍所有点都没有赢就平局，猫鼠一共2*len
    	if (x == y) return dp[t][x][y] = 2; // 猫鼠同一个位置猫赢
    	if (x == 0) return dp[t][x][y] = 1; // 鼠在位置0鼠赢
    	if (dp[t][x][y] != -1) return dp[t][x][y];
    	bool mouseTurn = (t % 2 == 0); // 步数为偶数老鼠走，步数为奇数猫走
    	if (mouseTurn) {
    		bool catWin = true;
    		for (int i = 0; i < graph[x].size(); ++i) { // 遍历老鼠可以走的下一个结点
    			int next = helper(graph, t + 1, graph[x][i], y, dp);
    			if (next == 1) return dp[t][x][y] = 1; // 鼠赢
    			else if (next != 2) catWin = false; // 猫没赢
    		}
    		if (catWin) return dp[t][x][y] = 2; // 猫赢
    		else return dp[t][x][y] = 0; // 陷入僵局
    	} else {
    		bool mouseWin = true;
    		for (int i = 0; i < graph[y].size(); ++i) {
    			if (graph[y][i] == 0) continue; // 猫不进鼠洞
    			int next = helper(graph, t + 1, x, graph[y][i], dp);
    			if (next == 2) return dp[t][x][y] = 2;
    			else if (next != 1) mouseWin = false;
    		}
    		if (mouseWin) return dp[t][x][y] = 1;
    		else return dp[t][x][y] = 0;
    	}
    }
    int catMouseGame(vector<vector<int>>& graph) {
        int len = graph.size();
        vector<vector<vector<int>>> dp(2*len, vector<vector<int>>(len, vector<int>(len, -1))); // 初始化为-1
        return helper(graph, 0, 1, 2, dp);
    }
};
```

## 排序

### [242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

难度：简单 # 2020.11.30

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

**示例 1:**
```
输入: s = "anagram", t = "nagaram"
输出: true
```
**示例 2:**
```
输入: s = "rat", t = "car"
输出: false
```
**说明:**
你可以假设字符串只包含小写字母。

**进阶:**
如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        int slen = s.size(), tlen = t.size();
        if(slen != tlen) return false;
        vector<vector<int>> m(2, vector<int>(26, 0));
        for(int i=0; i<slen; i++){
            m[0][s[i]-'a']++;
            m[1][t[i]-'a']++;
        }
        for(int i=0; i<26; i++){
            if(m[0][i] != m[1][i]) return false;
        }
        return true;
    }
};
```

### [922. 按奇偶排序数组 II](https://leetcode-cn.com/problems/sort-array-by-parity-ii/)

难度：简单 # 2020.11.30

给定一个非负整数数组 `A`， A 中一半整数是奇数，一半整数是偶数。

对数组进行排序，以便当 `A[i]` 为奇数时，`i` 也是奇数；当 `A[i]` 为偶数时， `i` 也是偶数。

你可以返回任何满足上述条件的数组作为答案。

**示例：**
```
输入：[4,2,5,7]
输出：[4,5,2,7]
解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
```
**提示：**
+ `2 <= A.length <= 20000`
+ `A.length % 2 == 0`
+ `0 <= A[i] <= 1000`

```cpp
class Solution {
public:
    vector<int> sortArrayByParityII(vector<int>& A) {
        int len = A.size();
        vector<int> odd, even, ans;
        for(int i=0; i<len; i++){
            if(A[i]%2 == 0) even.push_back(A[i]);
            else odd.push_back(A[i]);
        }
        for(int i=0; i<len/2; i++){
            ans.push_back(even[i]);
            ans.push_back(odd[i]);
        }
        return ans;
    }
};
```

### [1329. 将矩阵按对角线排序](https://leetcode-cn.com/problems/sort-the-matrix-diagonally/)

难度：中等 # 2020.11.30

给你一个 `m * n` 的整数矩阵 `mat` ，请你将同一条对角线上的元素（从左上到右下）按升序排序后，返回排好序的矩阵。

**示例 1：**

![1482_example_1_2](./images/1482_example_1_2.png)
```
输入：mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
输出：[[1,1,1,1],[1,2,2,2],[1,2,3,3]]
```
**提示：**
+ `m == mat.length`
+ `n == mat[i].length`
+ `1 <= m, n <= 100`
+ `1 <= mat[i][j] <= 100`

```cpp
class Solution {
public:
    vector<vector<int>> diagonalSort(vector<vector<int>>& mat) {
        int r = mat.size();
        int c = mat[0].size();
        vector<vector<int>> vlist(r+c-1, vector<int>());
        for(int i=0; i<r-1; i++){ // 左侧一列从下到上（排除[0][0]）
            int m = r-i-1;
            int n = 0;
            while(m < r && n < c){
                vlist[i].push_back(mat[m][n]);
                m++; n++;
            }
            sort(vlist[i].begin(), vlist[i].end());
            m = r-i-1;
            n = 0;
            int cnt = 0;
            while(m < r && n < c){
                mat[m][n] = vlist[i][cnt];
                m++; n++; cnt++;
            }
        }
        for(int i=r-1; i<r+c-1; i++){ // 上侧一行从左到右（包括[0][0]）
            int m = 0;
            int n = i-r+1;
            while(m < r && n < c){
                vlist[i].push_back(mat[m][n]);
                m++; n++;
            }
            sort(vlist[i].begin(), vlist[i].end());
            m = 0;
            n = i-r+1;
            int cnt = 0;
            while(m < r && n < c){
                mat[m][n] = vlist[i][cnt];
                m++; n++; cnt++;
            }
        }
        return mat;
    }
};
```

### [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

难度：中等

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

**示例 1:**
```
输入: [10,2]
输出: "102"
```
**示例 2:**
```
输入: [3,30,34,5,9]
输出: "3033459"
```
**提示:**
+ `0 < nums.length <= 100`
**说明:**
+ 输出结果可能非常大，所以你需要返回一个字符串而不是整数
+ 拼接起来的数字可能会有前导 0，最后结果不需要去掉前导 0

```cpp
class Solution {
public:
    static bool cmp(int a, int b){
        string sa = to_string(a);
        string sb = to_string(b);
        string sasb = sa + sb;
        string sbsa = sb + sa;
        int len = sasb.size();
        for(int i=0; i<len; i++){
            if(sasb[i] != sbsa[i])
                return sasb[i] < sbsa[i];
        }
        return false;
    }
    string minNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), cmp);
        string ans;
        int len = nums.size();
        
        for(int i=0; i<len; i++){
            ans += to_string(nums[i]);
        }
        return ans;
    }
};
```

### [面试题 17.14. 最小K个数](https://leetcode-cn.com/problems/smallest-k-lcci/)

难度：中等 # 2020.11.30

设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可。

**示例：**
```
输入： arr = [1,3,5,7,2,4,6,8], k = 4
输出： [1,2,3,4]
```
**提示：**
+ `0 <= len(arr) <= 100000`
+ `0 <= k <= min(100000, len(arr))`

```cpp
class Solution {
public:
    // void quick_sort(vector<int>& arr, int left, int right){ // 快排函数
    //     if(left >= right){
    //         return;
    //     }
    //     int key = arr[left];
    //     int l = left;
    //     int r = right;
    //     while(l < r){
    //         while(l < r && arr[r] >= key){
    //             r--;
    //         }
    //         if(l < r){
    //             arr[l] = arr[r];
    //             l++;
    //         }
    //         while(l < r && arr[l] < key){
    //             l++;
    //         }
    //         if(l < r){
    //             arr[r] = arr[l];
    //             r--;
    //         }
    //     }
    //     arr[l] = key;
    //     quick_sort(arr, left, l-1);
    //     quick_sort(arr, l+1, right);
    //     return;
    // }
    // void QuickSort(int* a, int left, int right){ // 更简洁的快排
    //     if (left < right) {
    //         int l = left;
    //         int r = right;
    //         int tmp = a[l];
    //         while (l != r){
    //             while (l < r&&a[r] >= tmp)
    //                 r--;
    //             while (l < r&&a[l] <= tmp)
    //                 l++;
    //             swap(a[l], a[r]); 
    //         }
    //         swap(a[l], a[left]);
    //         QuickSort(a, left, l-1);
    //         QuickSort(a, l + 1, right);
    //     }
    // }
    int quick_sort_once(vector<int>& arr, int left, int right){
        int key = arr[left];
        int l = left;
        int r = right;
        while(l < r){
            while(l < r && arr[r] >= key){
                r--;
            }
            if(l < r){
                arr[l] = arr[r];
                l++;
            }
            while(l < r && arr[l] < key){
                l++;
            }
            if(l < r){
                arr[r] = arr[l];
                r--;
            }
        }
        arr[l] = key;
        return l;
    }
    vector<int> smallestK(vector<int>& arr, int k) {
        int len = arr.size();
        int left=0, right=len-1;
        while(1){
            if(left>=right || left<0 || right<0) break;
            int idx = quick_sort_once(arr, left, right);
            if(idx > k){
                right = idx-1;
            }
            else if(idx < k){
                left = idx+1;
            }
            else break;
        }
        vector<int> ans;
        for(int i=0; i<k; i++){
            ans.push_back(arr[i]);
        }
        return ans;
    }
};
```

### [1630. 等差子数组](https://leetcode-cn.com/problems/arithmetic-subarrays/)

难度：中等 # 2020.12.01

如果一个数列由至少两个元素组成，且每两个连续元素之间的差值都相同，那么这个序列就是 **等差数列**。更正式地，数列 `s` 是等差数列，只需要满足：对于每个有效的` i` ， `s[i+1] - s[i] == s[1] - s[0]` 都成立。

例如，下面这些都是 **等差数列** ：
```
1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9
```
下面的数列 **不是等差数列** ：
```
1, 1, 2, 5, 7
```
给你一个由 `n` 个整数组成的数组 `nums`，和两个由 `m` 个整数组成的数组 `l` 和 `r`，后两个数组表示 `m` 组范围查询，其中第 `i` 个查询对应范围 `[l[i], r[i]]` 。所有数组的下标都是 **从 0 开始** 的。

返回 `boolean` 元素构成的答案列表 `answer` 。如果子数组 `nums[l[i]], nums[l[i]+1], ... , nums[r[i]]` 可以 **重新排列** 形成 **等差数列** ，`answer[i]` 的值就是 `true`；否则 `answer[i]` 的值就是 `false`。

**示例 1：**

```
输入：nums = [4,6,5,9,3,7], l = [0,0,2], r = [2,3,5]
输出：[true,false,true]
解释：
第 0 个查询，对应子数组 [4,6,5] 。可以重新排列为等差数列 [6,5,4] 。
第 1 个查询，对应子数组 [4,6,5,9] 。无法重新排列形成等差数列。
第 2 个查询，对应子数组 [5,9,3,7] 。可以重新排列为等差数列 [3,5,7,9] 。
```
**示例 2：**

```
输入：nums = [-12,-9,-3,-12,-6,15,20,-25,-20,-15,-10], l = [0,1,6,4,8,7], r = [4,4,9,7,9,10]
输出：[false,true,false,false,true,true]
```

**提示：**

+ `n == nums.length`
+ `m == l.length`
+ `m == r.length`
+ `2 <= n <= 500`
+ `1 <= m <= 500`
+ `0 <= l[i] < r[i] < n`
+ `-105 <= nums[i] <= 105`

```cpp
class Solution {
public:
    bool check(vector<int>& nums, int left, int right){
        vector<int> v{&nums[0]+left, &nums[0]+right+1}; // 高级列表分段方法
        int len = v.size();
        if(len == 1) return true;
        sort(v.begin(), v.end());
        int d = v[1] - v[0];
        for(int i=1; i<len; i++){
            if(v[i] - v[i-1] != d) return false;
        }
        return true;
    }
    vector<bool> checkArithmeticSubarrays(vector<int>& nums, vector<int>& l, vector<int>& r) {
        vector<bool> ans;
        int len = l.size();
        for(int i=0; i<len; i++){
            ans.push_back(check(nums, l[i], r[i]));
        }
        return ans;
    }
};
```

### [973. 最接近原点的 K 个点](https://leetcode-cn.com/problems/k-closest-points-to-origin/)

难度：中等 # 2020.12.01

我们有一个由平面上的点组成的列表 `points`。需要从中找出 `K` 个距离原点 `(0, 0)` 最近的点。

（这里，平面上两点之间的距离是欧几里德距离。）

你可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。

**示例 1：**
```
输入：points = [[1,3],[-2,2]], K = 1
输出：[[-2,2]]
解释： 
(1, 3) 和原点之间的距离为 sqrt(10)，
(-2, 2) 和原点之间的距离为 sqrt(8)，
由于 sqrt(8) < sqrt(10)，(-2, 2) 离原点更近。
我们只需要距离原点最近的 K = 1 个点，所以答案就是 [[-2,2]]。
```
**示例 2：**

```
输入：points = [[3,3],[5,-1],[-2,4]], K = 2
输出：[[3,3],[-2,4]]
（答案 [[-2,4],[3,3]] 也会被接受。）
```

**提示：**

+ `1 <= K <= points.length <= 10000`
+ `-10000 < points[i][0] < 10000`
+ `-10000 < points[i][1] < 10000`

```cpp
class Solution {
public:
    int dist(vector<int> a){
        return a[0] * a[0] + a[1] * a[1];
    }
    int quick_sort_once(vector<vector<int>>& points, int left, int right){
        vector<int> key = points[left];
        int l = left;
        int r = right;
        while(l < r){
            while(l < r && dist(points[r]) >= dist(key)){
                r--;
            }
            while(l < r && dist(points[l]) <= dist(key)){
                l++;
            }
            swap(points[l], points[r]);
        }
        swap(points[l], points[left]);
        return l;
    }
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        int len = points.size();
        int left=0, right=len-1;
        while(1){
            if(left>=right || left<0 || right<0) break;
            int idx = quick_sort_once(points, left, right);
            if(idx > K){
                right = idx-1;
            }
            else if(idx < K){
                left = idx+1;
            }
            else break;
        }
        vector<vector<int>> ans;
        for(int i=0; i<K; i++){
            ans.push_back(points[i]);
        }
        return ans;
    }
};
```

### [179. 最大数](https://leetcode-cn.com/problems/largest-number/)

难度：中等 2021.04.12

给定一组非负整数 nums，重新排列每个数的顺序（每个数不可拆分）使之组成一个最大的整数。

注意：输出结果可能非常大，所以你需要返回一个字符串而不是整数。

**示例 1：**
```
输入：nums = [10,2]
输出："210"
```
**示例 2：**
```
输入：nums = [3,30,34,5,9]
输出："9534330"
```
**示例 3：**
```
输入：nums = [1]
输出："1"
```
**示例 4：**
```
输入：nums = [10]
输出："10"
```
**提示：**

+ `1 <= nums.length <= 100`
+ `0 <= nums[i] <= 10^9`

```cpp
class Solution {
public:
    static bool cmp(int a, int b){
        long a_len = 10, b_len = 10;
        while (a_len <= a) {
            a_len *= 10;
        }
        while (b_len <= b) {
            b_len *= 10;
        }
        return a_len * b + a < b_len * a + b;
    }
    string largestNumber(vector<int> &nums) {
        sort(nums.begin(), nums.end(), cmp);
        if (nums[0] == 0) {
            return "0";
        }
        string s;
        for (int &x : nums) {
            s += to_string(x);
        }
        return s;
    }
};
```

### [220. 存在重复元素 III](https://leetcode-cn.com/problems/contains-duplicate-iii/)

难度：中等 # 2021.04.17

给你一个整数数组 `nums` 和两个整数 `k` 和 `t` 。请你判断是否存在 **两个不同下标** `i` 和 `j`，使得 `abs(nums[i] - nums[j]) <= t` ，同时又满足 `abs(i - j) <= k` 。

如果存在则返回 `true`，不存在返回 `false`。

**示例 1：**
```
输入：nums = [1,2,3,1], k = 3, t = 0
输出：true
```
**示例 2：**
```
输入：nums = [1,0,1,1], k = 1, t = 2
输出：true
```
**示例 3：**
```
输入：nums = [1,5,9,1,5,9], k = 2, t = 3
输出：false
```

**提示：**

+ `0 <= nums.length <= 2 * 10^4`
+ `-2^31 <= nums[i] <= 2^31 - 1`
+ `0 <= k <= 10^4`
+ `0 <= t <= 2^31 - 1`

```cpp
class Solution {
public:
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        set<long> s;
        for(int i=0; i<nums.size(); i++){
            auto lb = s.lower_bound((long)nums[i] - t); // 查找第一个键值不小于key的元素的迭代器
            if(lb != s.end() && *lb <= (long)nums[i] + t) return 1;
            s.insert(nums[i]);
            if(i >= k) s.erase(nums[i - k]);
        }
        return 0;
    }
};
```



## 回溯

### [401. 二进制手表](https://leetcode-cn.com/problems/binary-watch/)

难度：简单 # 2020.12.07

二进制手表顶部有 4 个 LED 代表 **小时（0-11）**，底部的 6 个 LED 代表 **分钟（0-59）**。

每个 LED 代表一个 0 或 1，最低位在右侧。

<img src="./images/Binary_clock_samui_moon.jpg" alt="Binary_clock_samui_moon" style="zoom:25%;" />

例如，上面的二进制手表读取 “3:25”。

给定一个非负整数 n 代表当前 LED 亮着的数量，返回所有可能的时间。

**示例：**

```
输入: n = 1
返回: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
```
**提示：**

+ 输出的顺序没有要求。
+ 小时不会以零开头，比如 “01:00” 是不允许的，应为 “1:00”。
+ 分钟必须由两位数组成，可能会以零开头，比如 “10:2” 是无效的，应为 “10:02”。
+ 超过表示范围（**小时 0-11**，**分钟 0-59**）的数据将会被舍弃，也就是说不会出现 "13:00", "0:61" 等时间。

```cpp
class Solution { // 回溯法做的，实际上直接遍历0:00-11:59间位为1数量和为num更简单
public:
    string trans(vector<int>& bin){
        int hour = 8*bin[0] + 4*bin[1] + 2*bin[2] + bin[3];
        int minute = 32*bin[4] + 16*bin[5] + 8*bin[6] + 4*bin[7] + 2*bin[8] + bin[9];
        if(hour > 11 || minute > 59) return "";
        if(minute > 9) return to_string(hour) + ":" + to_string(minute);
        else return to_string(hour) + ":0" + to_string(minute);
    }
    void backtrack(int num, int start, vector<string>& ans, vector<int>& bin){
        if(num < 0) return;
        if(num == 0 && start == 10){
            string time = trans(bin);
            if(time != "")
                ans.push_back(time);
        }
        else{
            if(start < 10){
                backtrack(num, start+1, ans, bin);
                bin[start] = 1;
                backtrack(num-1, start+1, ans, bin);
                bin[start] = 0;
            }
        }
    }
    vector<string> readBinaryWatch(int num) {
        vector<string> ans;
        vector<int> bin(10, 0);
        backtrack(num, 0, ans, bin);
        return ans;
    }
};
```

### [面试题 08.04. 幂集](https://leetcode-cn.com/problems/power-set-lcci/)

难度：中等 # 2020.12.08

幂集。编写一种方法，返回某集合的所有子集。集合中**不包含重复的元素**。

说明：解集不能包含重复的子集。

**示例：**

```
 输入： nums = [1,2,3]
 输出：
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
```

```cpp
class Solution { // 回溯
public:
    void backtrack(vector<int>& nums, int idx, vector<int>& subset, vector<vector<int>>& ans) {
        if (idx >= nums.size()) {
            ans.push_back(subset);
            return;
        }
        backtrack(nums, idx+1, subset, ans); // 不选
        subset.push_back(nums[idx]); // 选
        backtrack(nums, idx+1, subset, ans);
        subset.pop_back();
    }
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> subset;
        backtrack(nums, 0, subset, ans);
        return ans;
    }
};

class Solution { // 位运算
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        vector<int> subset;
        int len = nums.size();
        for(int i=0; i<1<<len; i++){
            subset.clear();
            int bits = i;
            for(int cnt=0; cnt<len; cnt++){
                if((bits&1) == 1){
                    subset.push_back(nums[cnt]);
                }
                bits>>=1;
            }
            ans.push_back(subset);
        }
        return ans;
    }
};
```

### [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

难度：中等 # 2020.12.08

给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

**示例：**
```
输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]
```

```cpp
class Solution {
public:
    bool isPalindrome(string& s, int l, int r){ // 闭区间
        while(l < r){
            if(s[l] != s[r]) return false;
            l++; r--;
        }
        return true;
    }
    void backtrack(string& s, int pre, int idx, vector<string>& plan, vector<vector<string>>& ans){ // pre是新一段的开始，idx是当前位置
        if(idx == s.size()){
            if(pre == s.size()) ans.push_back(plan);
            return;
        }
        if(isPalindrome(s, pre, idx)){
            plan.push_back(s.substr(pre, idx-pre+1));
            backtrack(s, idx+1, idx+1, plan, ans);
            plan.pop_back();
        }
        backtrack(s, pre, idx+1, plan, ans);
    }
    vector<vector<string>> partition(string s) {
        int len = s.size();
        vector<vector<string>> ans;
        vector<string> plan;
        backtrack(s, 0, 0, plan, ans);
        return ans;
    }
};
```

### [526. 优美的排列](https://leetcode-cn.com/problems/beautiful-arrangement/)

难度：中等 # 2020.12.26

假设有从 1 到 **N** 的 N 个整数，如果从这 **N** 个数字中成功构造出一个数组，使得数组的第 **i** 位 (1 <= i <= N) 满足如下两个条件中的一个，我们就称这个数组为一个优美的排列。条件：

第 **i** 位的数字能被 **i** 整除
**i** 能被第 **i** 位上的数字整除
现在给定一个整数 N，请问可以构造多少个优美的排列？

**示例1：**
```
输入: 2
输出: 2
解释: 

第 1 个优美的排列是 [1, 2]:
  第 1 个位置（i=1）上的数字是1，1能被 i（i=1）整除
  第 2 个位置（i=2）上的数字是2，2能被 i（i=2）整除

第 2 个优美的排列是 [2, 1]:
  第 1 个位置（i=1）上的数字是2，2能被 i（i=1）整除
  第 2 个位置（i=2）上的数字是1，i（i=2）能被 1 整除
```
**说明：**

1. **N** 是一个正整数，并且不会超过15。

```cpp
class Solution { // 本质是全排列
public:
    void backtrack(const int& N, int depth, vector<int>& vis, int& cnt){
        if(depth == N){
            cnt++;
            return;
        }
        for(int i=1; i <= N; i++){
            if(!vis[i-1] && ((depth+1)%i==0 || i%(depth+1)==0)){
                vis[i-1] = 1;
                backtrack(N, depth+1, vis, cnt);
                vis[i-1] = 0;
            }
        }
    }
    int countArrangement(int N) {
        vector<int> vis(N, 0);
        int ans = 0;
        backtrack(N, 0, vis, ans);
        return ans;
    }
};
```



## 哈希表

### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

难度：简单 # 2020.12.14

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。

**示例：**
```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hashtable; // 值和索引的映射
        for (int i = 0; i < nums.size(); ++i) { // 哈希表，对于每个i，先看target-i在不在哈希表中，然后再将i加入哈希表，避免同一个元素用两次
            auto it = hashtable.find(target - nums[i]);
            if (it != hashtable.end()) {
                return {it->second, i};
            }
            hashtable[nums[i]] = i;
        }
        return {};
    }
};
```

### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

难度：简单 # 2020.12.14

给定一个**非**空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

**说明：**

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

**示例 1：**
```
输入: [2,2,1]
输出: 1
```
**示例 2：**
```
输入: [4,1,2,1,2]
输出: 4
```

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = nums[0];
        int len = nums.size();
        for(int i=1; i<len; i++){
            ans ^= nums[i];
        }
        return ans;
    }
};
```

### [202. 快乐数](https://leetcode-cn.com/problems/happy-number/)

难度：简单 # 2020.12.14

编写一个算法来判断一个数 `n` 是不是快乐数。

「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 **无限循环** 但始终变不到 1。如果 **可以变为**  1，那么这个数就是快乐数。

如果 `n` 是快乐数就返回 `True` ；不是，则返回 `False` 。

**示例：**
```
输入：19
输出：true
解释：
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
```

```cpp
class Solution {
public:
    bool isHappy(int n) {
        set<int> s;
        while(1){
            auto it = s.find(n);
            if(it != s.end()){
                return false;
            }
            if(n == 1) return true;
            s.insert(n);
            int newn = 0;
            while(n > 0){
                newn += pow(n%10, 2);
                n /= 10;
            }
            n = newn;
        }
    }
};
```

### [299. 猜数字游戏](https://leetcode-cn.com/problems/bulls-and-cows/)

难度：中等 # 2020.12.26

你在和朋友一起玩 猜数字（Bulls and Cows）游戏，该游戏规则如下：

1. 你写出一个秘密数字，并请朋友猜这个数字是多少。

2. 朋友每猜测一次，你就会给他一个提示，告诉他的猜测数字中有多少位属于数字和确切位置都猜对了（称为“Bulls”, 公牛），有多少位属于数字猜对了但是位置不对（称为“Cows”, 奶牛）。

3. 朋友根据提示继续猜，直到猜出秘密数字。

请写出一个根据秘密数字和朋友的猜测数返回提示的函数，返回字符串的格式为 `xAyB` ，`x` 和 `y` 都是数字，`A` 表示公牛，用 `B` 表示奶牛。

+ `xA` 表示有 `x` 位数字出现在秘密数字中，且位置都与秘密数字一致。
+ `yB` 表示有 `y` 位数字出现在秘密数字中，但位置与秘密数字不一致。
请注意秘密数字和朋友的猜测数都可能含有重复数字，每位数字只能统计一次。

**示例 1：**
```
输入: secret = "1807", guess = "7810"
输出: "1A3B"
解释: 1 公牛和 3 奶牛。公牛是 8，奶牛是 0, 1 和 7。
```
**示例 2：**
```
输入: secret = "1123", guess = "0111"
输出: "1A1B"
解释: 朋友猜测数中的第一个 1 是公牛，第二个或第三个 1 可被视为奶牛。
```

**说明：**你可以假设秘密数字和朋友的猜测数都只包含数字，并且它们的长度永远相等。

```cpp
class Solution {
public:
    string getHint(string secret, string guess) {
        int len = secret.size();
        string s, g;
        int A = 0, B = 0;
        for(int i=0; i<len; i++){
            if(secret[i] != guess[i]){
                s += secret[i];
                g += guess[i];
            }else A++;
        }
        vector<int> s_cnt(10, 0);
        vector<int> g_cnt(10, 0);
        for(int i=0; i<len-A; i++){ // 统计每个数字出现多少次
            s_cnt[s[i]-'0']++;
            g_cnt[g[i]-'0']++;
        }
        for(int i=0; i<10; i++){
            B += min(s_cnt[i], g_cnt[i]);
        }
        return to_string(A)+"A"+to_string(B)+"B";
    }
};
```

### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

难度：中等 # 2020.12.31

给定一个非空的整数数组，返回其中出现频率前 **k** 高的元素。

**示例 1：**
```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```
**示例 2：**
```
输入: nums = [1], k = 1
输出: [1]
```
**提示：**

+ 你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
+ 你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小。
+ 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。
+ 你可以按任意顺序返回答案。

```cpp
class Solution {
public:
    struct cmp{
        bool operator()(const pair<int, int>& a, const pair<int, int>& b){
            return a.second > b.second;
        }
    }; 
    vector<int> topKFrequent(vector<int>& nums, int k) {
        int len = nums.size();
        unordered_map<int, int> cnt;
        for(int i=0; i<len; i++){
            cnt[nums[i]]++;
        }
        priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> q; // 小根堆 C++优先队列默认是大根堆
        for (auto& [num, count] : cnt) {
            if (q.size() == k) {
                if (q.top().second < count) {
                    q.pop();
                    q.emplace(num, count); // 原地构造一个元素并插入队列
                }
            } else {
                q.emplace(num, count);
            }
        }
        vector<int> ans;
        while (!q.empty()) {
            ans.emplace_back(q.top().first); // C++11中比push_back更强大
            q.pop();
        }
        return ans;
    }
};
// 建立一个小顶堆，然后遍历「出现次数数组」
// 如果堆的元素个数小于 k，就可以直接插入堆中。
// 如果堆的元素个数等于 k，则检查堆顶与当前出现次数的大小。如果堆顶更大，说明至少有 k 个数字的出现次数比当前值大，故舍弃当前值；否则，就弹出堆顶，并将当前值插入堆中。
// 遍历完成后，堆中的元素就代表了「出现次数数组」中前 k 大的值。
```

### [560. 和为K的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

难度：中等 # 2020.12.25

给定一个整数数组和一个整数 **k**，你需要找到该数组中和为 **k** 的连续的子数组的个数。

**示例 1：**
```
输入:nums = [1,1,1], k = 2
输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。
```
**说明：**

1. 数组的长度为 [1, 20,000]。
2. 数组中元素的范围是 [-1000, 1000] ，且整数 **k** 的范围是 [-1e7, 1e7]。

```cpp
class Solution { // 暴力 累计和之差为连续和
public:
    int subarraySum(vector<int>& nums, int k) {
        int size = nums.size();
        vector<int> sums(size, 0);
        sums[0] = nums[0];
        for(int i=1; i<size; i++){
            sums[i] = sums[i-1] + nums[i];
        }
        int ans = 0;
        for(int i=0; i<size; i++){
            for(int j=i; j<size; j++){
                if(i == j){
                    if(sums[i] == k)
                        ans++;
                    continue;
                }
                else if(sums[j]-sums[i] == k)
                    ans++;
            }
        }
        return ans;
    }
};
```

### [1128. 等价多米诺骨牌对的数量](https://leetcode-cn.com/problems/number-of-equivalent-domino-pairs/)

难度：简单 # 2021.01.26

给你一个由一些多米诺骨牌组成的列表 `dominoes`。

如果其中某一张多米诺骨牌可以通过旋转 `0` 度或 `180` 度得到另一张多米诺骨牌，我们就认为这两张牌是等价的。

形式上，`dominoes[i] = [a, b]` 和 `dominoes[j] = [c, d]` 等价的前提是 `a==c` 且 `b==d`，或是 `a==d` 且 `b==c`。

在 `0 <= i < j < dominoes.length` 的前提下，找出满足 `dominoes[i]` 和 `dominoes[j]` 等价的骨牌对 `(i, j)` 的数量。

**示例：**

```
输入：dominoes = [[1,2],[2,1],[3,4],[5,6]]
输出：1
```

**提示：**

- `1 <= dominoes.length <= 40000`
- `1 <= dominoes[i][j] <= 9`

```cpp
class Solution {
public:
    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        int len = dominoes.size();
        unordered_map<int, int> cnt;
        for(auto& it : dominoes){
            it[0] < it[1] ? cnt[10*it[0]+it[1]]++ : cnt[10*it[1]+it[0]]++;
        }
        int ans = 0;
        for(auto& [k, v] : cnt){
            ans += v*(v-1)/2;
        }
        return ans;
    }
};
```

### [705. 设计哈希集合](https://leetcode-cn.com/problems/design-hashset/)

难度：简单 # 2021.03.13

不使用任何内建的哈希表库设计一个哈希集合（HashSet）。

实现 `MyHashSet` 类：

+ `void add(key)` 向哈希集合中插入值 `key`。
+ `bool contains(key)` 返回哈希集合中是否存在这个值 `key`。
+ `void remove(key)` 将给定值 `key` 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。

**示例：**
```
输入：
["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
输出：
[null, null, null, true, false, null, true, null, false]

解释：
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // 返回 True
myHashSet.contains(3); // 返回 False ，（未找到）
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // 返回 True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // 返回 False ，（已移除）
```

**提示：**

+ `0 <= key <= 10^6`
+ 最多调用 `10^4` 次 `add`、`remove` 和 `contains`。

**进阶：**你可以不使用内建的哈希集合库解决此问题吗？

```cpp
class MyHashSet {
public:
    set<int> s;
    /** Initialize your data structure here. */
    MyHashSet() {
        s.clear();
    }
    
    void add(int key) {
        s.insert(key);
    }
    
    void remove(int key) {
        s.erase(key);
    }
    
    /** Returns true if this set contains the specified element */
    bool contains(int key) {
        if(s.find(key) == s.end()) return false;
        return true;
    }
};

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */
```

### [706. 设计哈希映射](https://leetcode-cn.com/problems/design-hashmap/)

难度：简单 # 2021.03.14

不使用任何内建的哈希表库设计一个哈希映射（HashMap）。

实现 `MyHashMap` 类：

+ `MyHashMap()` 用空映射初始化对象
+ `void put(int key, int value)` 向 `HashMap` 插入一个键值对 `(key, value)`。如果 `key` 已经存在于映射中，则更新其对应的值 `value`。
+ `int get(int key)` 返回特定的 `key` 所映射的 `value` ；如果映射中不包含 `key` 的映射，返回 `-1`。
+ `void remove(key)` 如果映射中存在 `key` 的映射，则移除 `key` 和它所对应的 `value`。

**示例：**
```
输入：
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
输出：
[null, null, null, 1, -1, null, 1, null, -1]

解释：
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // myHashMap 现在为 [[1,1]]
myHashMap.put(2, 2); // myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(1);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(3);    // 返回 -1（未找到），myHashMap 现在为 [[1,1], [2,2]]
myHashMap.put(2, 1); // myHashMap 现在为 [[1,1], [2,1]]（更新已有的值）
myHashMap.get(2);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,1]]
myHashMap.remove(2); // 删除键为 2 的数据，myHashMap 现在为 [[1,1]]
myHashMap.get(2);    // 返回 -1（未找到），myHashMap 现在为 [[1,1]]
```

**提示：**

+ `0 <= key, value <= 10^6`
+ 最多调用 `10^4` 次 `put`、`get` 和 `remove` 方法

**进阶：**你能否不使用内置的 HashMap 库解决此问题？

```cpp
class MyHashMap {
public:
    vector<pair<int, int>> hp;
    /** Initialize your data structure here. */
    MyHashMap() {
        vector<pair<int, int>>().swap(hp);
    }
    
    /** value will always be non-negative. */
    void put(int key, int value) {
        int len = hp.size();
        int i;
        for(i=0; i<len; i++){
            if(hp[i].first == key){
                hp[i].second = value;
                return;
            }
        }
        if(i == len) hp.push_back(make_pair(key, value));
    }
    
    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    int get(int key) {
        for(auto& it: hp){
            if(it.first == key) return it.second;
        }
        return -1;
    }
    
    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    void remove(int key) {
        int len = hp.size();
        for(auto& it: hp){
            if(it.first == key){
                pair<int, int> tmp = hp[len-1];
                hp[len-1] = it;
                it = tmp;
                hp.pop_back();
                return;
            }
        }
    }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */
```



## 栈

### [1544. 整理字符串](https://leetcode-cn.com/problems/make-the-string-great/)

难度：简单 # 2020.12.24

给你一个由大小写英文字母组成的字符串 s 。

一个整理好的字符串中，两个相邻字符 `s[i]` 和 `s[i+1]`，其中 `0<= i <= s.length-2` ，要满足如下条件:

+ 若 `s[i]` 是小写字符，则 `s[i+1]` 不可以是相同的大写字符。
+ 若 `s[i]` 是大写字符，则 `s[i+1]` 不可以是相同的小写字符。

请你将字符串整理好，每次你都可以从字符串中选出满足上述条件的 **两个相邻** 字符并删除，直到字符串整理好为止。

请返回整理好的 **字符串** 。题目保证在给出的约束条件下，测试样例对应的答案是唯一的。

**注意：**空字符串也属于整理好的字符串，尽管其中没有任何字符。


**示例 1：**
```
输入：s = "leEeetcode"
输出："leetcode"
解释：无论你第一次选的是 i = 1 还是 i = 2，都会使 "leEeetcode" 缩减为 "leetcode" 。
```
**示例 2：**
```
输入：s = "abBAcC"
输出：""
解释：存在多种不同情况，但所有的情况都会导致相同的结果。例如：
"abBAcC" --> "aAcC" --> "cC" --> ""
"abBAcC" --> "abBA" --> "aA" --> ""
```
**示例 3：**
```
输入：s = "s"
输出："s"
```

**提示：**

+ `1 <= s.length <= 100`
+ `s` 只包含小写和大写英文字母

```cpp
class Solution {
public:
    string stack2str(stack<char> S){
        int len = S.size();
        string ans;
        ans.resize(len);
        for(int i=0; i<len; i++){
            ans[len-i-1] = S.top(); S.pop();
        }
        return ans;
    }
    string makeGood(string s) {
        int len = INT_MAX, n = s.size();
        stack<char> S;
        while(n < len){
            stack<char>().swap(S);
            for(int i=0; i<n; i++){
                if(S.empty()){
                    S.push(s[i]); continue;
                }
                if(abs(S.top() - s[i]) == 32){
                    S.pop();
                } else {
                    S.push(s[i]);
                }
            }
            s = stack2str(S);
            len = n;
            n = S.size();
        }
        return s;
    }
};
```

### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

难度：简单 # 2020.12.25

请你仅使用两个栈实现先入先出队列。队列应当支持一般队列的支持的所有操作（`push`、`pop`、`peek`、`empty`）：

实现 `MyQueue` 类：

+ `void push(int x)` 将元素 x 推到队列的末尾
+ `int pop()` 从队列的开头移除并返回元素
+ `int peek()` 返回队列开头的元素
+ `boolean empty()` 如果队列为空，返回 `true` ；否则，返回 `false`


说明：

+ 你只能使用标准的栈操作 —— 也就是只有 `push to top`, `peek/pop from top`, `size`, 和 `is empty` 操作是合法的。
+ 你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。


进阶：

+ 你能否实现每个操作均摊时间复杂度为 `O(1)` 的队列？换句话说，执行 `n` 个操作的总时间复杂度为 `O(n)` ，即使其中一个操作可能花费较长时间。


**示例：**
```
输入：
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
输出：
[null, null, null, 1, 1, false]

解释：
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```

**提示：**

+ `1 <= x <= 9`
+ 最多调用 100 次 `push`、`pop`、`peek` 和 `empty`
+ 假设所有操作都是有效的 （例如，一个空的队列不会调用 `pop` 或者 `peek` 操作）

```cpp
class MyQueue {
public:
    stack<int> stool;
    stack<int> s;
    /** Initialize your data structure here. */
    MyQueue() {
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        s.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        while(!s.empty()){
            int t = s.top(); s.pop();
            stool.push(t);
        }
        int ans;
        if(!stool.empty()){
            ans = stool.top(); stool.pop();
        }
        while(!stool.empty()){
            int t = stool.top(); stool.pop();
            s.push(t);
        }
        return ans;
    }
    
    /** Get the front element. */
    int peek() {
        while(!s.empty()){
            int t = s.top(); s.pop();
            stool.push(t);
        }
        int ans;
        if(!stool.empty()){
            ans = stool.top();
        }
        while(!stool.empty()){
            int t = stool.top(); stool.pop();
            s.push(t);
        }
        return ans;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        if(s.empty()) return true;
        return false;
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

### [856. 括号的分数](https://leetcode-cn.com/problems/score-of-parentheses/)

难度：中等 # 2020.12.25

给定一个平衡括号字符串 S，按下述规则计算该字符串的分数：

+ `()` 得 1 分。
+ `AB` 得 `A + B` 分，其中 A 和 B 是平衡括号字符串。
+ `(A)` 得 `2 * A` 分，其中 A 是平衡括号字符串。

**示例 1：**
```
输入： "()"
输出： 1
```
**示例 2：**
```
输入： "(())"
输出： 2
```
**示例 3：**
```
输入： "()()"
输出： 2
```
**示例 4：**
```
输入： "(()(()))"
输出： 6
```

**提示：**

1. `S` 是平衡括号字符串，且只含有 `(` 和 `)` 。
2. `2 <= S.length <= 50`

```cpp
class Solution {
public:
    int scoreOfParentheses(string S) {
        vector<int> vS;
        int len = S.size();
        for(int i=0; i<len; i++){ // '(' -> -1 && ')' -> -2
            if(S[i] == '(') vS.push_back(-1); else vS.push_back(-2);
        }
        stack<int> s;
        
        for(int i=0; i<len; i++){
            if(s.empty()){
                s.push(vS[i]);
                continue;
            }
            int t = s.top();
            if(t == -1 && vS[i] == -2){ // 两种情况需要处理
                s.pop();
                int pending = 1;
                while(!s.empty()){
                    if(s.top() < 0) break;
                    else{
                        pending += s.top();
                        s.pop();
                    }
                }
                s.push(pending);
            }
            else if(t > 0 && vS[i] == -2){ // 两种情况需要处理
                int pending = s.top() * 2;
                s.pop(); s.pop();
                while(!s.empty()){
                    if(s.top() < 0) break;
                    else{
                        pending += s.top();
                        s.pop();
                    }
                }
                s.push(pending);
            }
            else s.push(vS[i]); // 其他情况直接添加
        }

        // cout<<"stack size: "<<s.size()<<endl;

        return s.top();
    }
};
```

### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

难度：中等 # 2020.12.25

请根据每日 `气温` 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]`，你的输出应该是 `[1, 1, 4, 2, 1, 1, 0, 0]`。

提示：`气温` 列表长度的范围是 `[1, 30000]`。每个气温的值的均为华氏度，都是在 `[30, 100]` 范围内的整数。

```cpp
class Solution { // 单调栈
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        int n = T.size();
        vector<int> ans(n);
        stack<int> s;
        for (int i = 0; i < n; ++i) {
            while (!s.empty() && T[i] > T[s.top()]) {
                int previousIndex = s.top();
                ans[previousIndex] = i - previousIndex;
                s.pop();
            }
            s.push(i);
        }
        return ans;
    }
};

class Solution { // 优化的暴力
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        vector<int> ans;
        int size = T.size();
        ans.resize(size);
        ans[size-1] = 0;
        for(int i=size-2; i>=0; i--){
            if(T[i] >= T[i+1]){
                if(ans[i+1] == 0)
                    ans[i] = 0;
                else{
                    int j;
                    for(j=i+ans[i+1]+1; j<size; j++)
                        if(T[j] > T[i]){
                            ans[i] = j -i;
                            break;
                        }
                    if(j == size)
                        ans[i] = 0;
                }
            }
            else
                ans[i] = 1;
        }
        return ans;
    }
};
```

### [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)

难度：中等 # 2020.12.26

给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

**示例 1：**
```
输入: [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；
数字 2 找不到下一个更大的数； 
第二个 1 的下一个最大的数需要循环搜索，结果也是 2。
```
**注意：**输入数组的长度不会超过 10000。

```cpp
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int len = nums.size();
        stack<int> s;
        vector<int> ans(len, -1);
        for(int i=0; i<len; i++){ // 第一次遍历得到一个单调栈（栈顶最小）
            if(!s.empty() && nums[s.top()] < nums[i]){
                while(!s.empty()){
                    if (nums[s.top()] < nums[i]){
                        int idx = s.top(); s.pop();
                        ans[idx] = nums[i];
                    } else break;
                }
            }
            s.push(i);
        }
        for(int i=0; i<len; i++){ // 第二次遍历解决循环数组问题
            if(s.size() == 1 && nums[i] == s.top()) break; // 仅剩最大值时
            if(nums[s.top()] < nums[i]){
                while(!s.empty()){
                    if (nums[s.top()] < nums[i]){
                        int idx = s.top(); s.pop();
                        ans[idx] = nums[i];
                    } else break;
                }
                if(nums[i] == s.top()) break;
                s.push(i);
            }
        }
        return ans;
    }
};
```

### [921. 使括号有效的最少添加](https://leetcode-cn.com/problems/minimum-add-to-make-parentheses-valid/)

给定一个由 `'('` 和 `')'` 括号组成的字符串 `S`，我们需要添加最少的括号（ `'('` 或是 `')'`，可以在任何位置），以使得到的括号字符串有效。

从形式上讲，只有满足下面几点之一，括号字符串才是有效的：

+ 它是一个空字符串，或者
+ 它可以被写成 `AB` （`A` 与 `B` 连接）, 其中 `A` 和 `B` 都是有效字符串，或者
+ 它可以被写作 `(A)`，其中 `A` 是有效字符串。

给定一个括号字符串，返回为使结果字符串有效而必须添加的最少括号数。

**示例 1：**
```
输入："())"
输出：1
```
**示例 2：**

```
输入："((("
输出：3
```
**示例 3：**

```
输入："()"
输出：0
```
**示例 4：**

```
输入："()))(("
输出：4
```

**提示：**

1. `S.length <= 1000`
2. `S` 只包含 `'('` 和 `')'` 字符。

```cpp
class Solution {
public:
    int minAddToMakeValid(string S) {
        int len = S.size();
        stack<char> s;
        for(int i=0; i<len; i++){
            if(S[i] == ')' && !s.empty() && s.top() == '('){
                s.pop();
            } else {
                s.push(S[i]);
            }
        }
        return s.size();
    }
};
```



### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

难度：困难 # 2020.12.27

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![histogram](./images/histogram.png)

以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 `[2,1,5,6,2,3]`。

![histogram_area](./images/histogram_area.png)

图中阴影部分为所能勾勒出的最大矩形面积，其面积为 `10` 个单位。

**示例：**

```
输入: [2,1,5,6,2,3]
输出: 10
```

```cpp
class Solution { // 单调栈
public:
    int largestRectangleArea(vector<int>& heights) {
        int len = heights.size() + 2;
        vector<int> h(len);
        for(int i=0; i<len-2; i++){
            h[i+1] = heights[i];
        }
        stack<int> s;
        int ans = 0;
        for(int i=0; i<len; i++){
            if(s.empty()) s.push(i);
            if(!s.empty()){
                while(h[s.top()] > h[i]){
                    int height = h[s.top()]; s.pop();
                    int idx_left = s.top();
                    // cout<<"idx_left: "<<idx_left<<" idx_right: "<<i<<" height: "<<height<<" size: "<<height * (i-idx_left-1)<<endl;
                    ans = max(ans, height * (i-idx_left-1));
                }
                s.push(i);
            }
        }
        return ans;
    }
};
```

### [85. 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

难度：困难 # 2020.12.30

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

**示例 1：**

![maximal](./images/maximal.jpg)

```
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
```
**示例 2：**
```
输入：matrix = []
输出：0
```
**示例 3：**
```
输入：matrix = [["0"]]
输出：0
```
**示例 4：**
```
输入：matrix = [["1"]]
输出：1
```
**示例 5：**
```
输入：matrix = [["0","0"]]
输出：0
```

**提示：**

+ `rows == matrix.length`
+ `cols == matrix[0].length`
+ `0 <= row, cols <= 200`
+ `matrix[i][j]` 为 `'0'` 或 `'1'`

```cpp
class Solution {
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int r = matrix.size();
        if(r == 0) return 0;
        int c = matrix[0].size();
        vector<int> heights(c+2, 0);
        int ans = 0;
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){ // 逐行利用84题的思想得到最大矩形
                matrix[i][j] == '0' ? heights[j+1] = 0 : heights[j+1] += matrix[i][j]-'0';
            }
            stack<int> s;
            for(int j=0; j<c+2; j++){
                if(s.empty()) s.push(j);
                if(!s.empty()){
                    while(heights[s.top()] > heights[j]){
                        int height = heights[s.top()]; s.pop();
                        int idx_left = s.top();
                        ans = max(ans, height * (j-idx_left-1));
                    }
                    s.push(j);
                }
            }
        }
        return ans;
    }
};
```

### [480. 滑动窗口中位数](https://leetcode-cn.com/problems/sliding-window-median/)

难度：困难 # 2021.02.03

中位数是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

例如：

+ `[2,3,4]`，中位数是 `3`
+ `[2,3]`，中位数是 `(2 + 3) / 2 = 2.5`

给你一个数组 *nums*，有一个长度为 *k* 的窗口从最左端滑动到最右端。窗口中有 *k* 个数，每次窗口向右移动 *1* 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。

**示例：**

给出 *nums* = `[1,3,-1,-3,5,3,6,7]`，以及 *k = 3*。
```
窗口位置                      中位数
---------------               -----
[1  3  -1] -3  5  3  6  7       1
 1 [3  -1  -3] 5  3  6  7      -1
 1  3 [-1  -3  5] 3  6  7      -1
 1  3  -1 [-3  5  3] 6  7       3
 1  3  -1  -3 [5  3  6] 7       5
 1  3  -1  -3  5 [3  6  7]      6
```
因此，返回该滑动窗口的中位数数组 [1,-1,-1,3,5,6]。

**提示：**

+ 你可以假设 `k` 始终有效，即：`k` 始终小于输入的非空数组的元素个数。
+ 与真实值误差在 `10 ^ -5` 以内的答案将被视作正确答案。

```cpp
class Solution {
public:
    multiset<int> ms;
    double getMid(int& k){
        if(k%2) return *next(ms.begin(), k/2);
        else return ((double)*(next(ms.begin(), k/2-1)) + (double)*next(ms.begin(), k/2)) * 0.5;
    }
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        for(int i=0; i<k-1; i++) ms.insert(nums[i]);
        vector<double> ans;
        for(int i=k-1; i<nums.size(); i++){
            ms.insert(nums[i]);
            ans.push_back(getMid(k));
            ms.erase(ms.find(nums[i-k+1]));
        }
        return ans;
    }
};
```

### [1047. 删除字符串中的所有相邻重复项](https://leetcode-cn.com/problems/remove-all-adjacent-duplicates-in-string/)

难度：简单 # 2021.03.09

给出由小写字母组成的字符串 `S`，**重复项删除操作**会选择两个相邻且相同的字母，并删除它们。

在 S 上反复执行重复项删除操作，直到无法继续删除。

在完成所有重复项删除操作后返回最终的字符串。答案保证唯一。

**示例：**
```
输入："abbaca"
输出："ca"
解释：
例如，在 "abbaca" 中，我们可以删除 "bb" 由于两字母相邻且相同，这是此时唯一可以执行删除操作的重复项。之后我们得到字符串 "aaca"，其中又只有 "aa" 可以执行重复项删除操作，所以最后的字符串为 "ca"。
```

**提示：**

+ `1 <= S.length <= 20000`
+ `S` 仅由小写英文字母组成。

```cpp
class Solution {
public:
    string removeDuplicates(string S) {
        stack<char> s;
        int len = S.size();
        for(int i=0; i<len; i++){
            if(!s.empty() && s.top() == S[i]) s.pop();
            else s.push(S[i]);
        }
        string ans;
        while(!s.empty()){
            ans += s.top(); s.pop();
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```

### [224. 基本计算器](https://leetcode-cn.com/problems/basic-calculator/)

难度：困难 # 2021.03.10

实现一个基本的计算器来计算一个简单的字符串表达式 `s` 的值。

**示例 1：**
```
输入：s = "1 + 1"
输出：2
```
**示例 2：**
```
输入：s = " 2-1 + 2 "
输出：3
```
**示例 3：**
```
输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23
```
**提示：**

- `1 <= s.length <= 3 * 105`
- `s` 由数字、`'+'`、`'-'`、`'('`、`')'`、和 `' '` 组成
- `s` 表示一个有效的表达式

```cpp
class Solution {
public:
    int calculate(string s) {
        stack<int> ops;
        ops.push(1);
        int sign = 1;

        int ans = 0;
        int n = s.length();
        int i = 0;
        while (i < n) {
            if (s[i] == ' ') {
                i++;
            } else if (s[i] == '+') {
                sign = ops.top();
                i++;
            } else if (s[i] == '-') {
                sign = -ops.top();
                i++;
            } else if (s[i] == '(') {
                ops.push(sign);
                i++;
            } else if (s[i] == ')') {
                ops.pop();
                i++;
            } else {
                long num = 0;
                while (i < n && s[i] >= '0' && s[i] <= '9') {
                    num = num * 10 + s[i] - '0';
                    i++;
                }
                ans += sign * num;
            }
        }
        return ans;
    }
};
```

### [227. 基本计算器 II](https://leetcode-cn.com/problems/basic-calculator-ii/)

难度：中等 # 2021.03.11

给你一个字符串表达式 `s` ，请你实现一个基本计算器来计算并返回它的值。

整数除法仅保留整数部分。

**示例 1：**

```
输入：s = "3+2*2"
输出：7
```

**示例 2：**

```
输入：s = " 3/2 "
输出：1
```

**示例 3：**

```
输入：s = " 3+5 / 2 "
输出：5
```

**提示：**

+ `1 <= s.length <= 3 * 10^5`
+ `s` 由整数和算符 `('+', '-', '*', '/')` 组成，中间由一些空格隔开
+ `s` 表示一个 **有效表达式**
+ 表达式中的所有整数都是非负整数，且在范围 `[0, 231 - 1]` 内
+ 题目数据保证答案是一个 **32-bit 整数**

```cpp
class Solution {
public:
    int calculate(string s) {
        int len = s.size();
        int ans = 0, tmp = 0;
        int i = 0;
        while(i < len){
            if(s[i] == ' '){
                i++;
            } else if(s[i] == '+'){
                ans += tmp; tmp = 0;
                i++;
            } else if(s[i] == '-'){
                ans += tmp; tmp = 0;
                i++;
            } else if(s[i] == '*'){
                i++;
            } else if(s[i] == '/'){
                i++;
            } else {
                int ii = i-1;
                long num = 0;
                while(i<len && s[i]>='0' && s[i]<='9'){
                    num = num * 10 + s[i] - '0';
                    i++;
                }
                while(ii>=0 && s[ii]==' ') ii--;
                if(ii >= 0){
                    if(s[ii] == '+'){
                        tmp += num;
                    } else if(s[ii] == '-'){
                        tmp -= num;
                    } else if(s[ii] == '*'){
                        tmp *= num;
                    } else if(s[ii] == '/'){
                        tmp /= num;
                    } else{
                        cout<<"Wrong branch!"<<endl;
                    }
                } else {
                    tmp += num;
                }
            }
        }
        ans += tmp;
        return ans;
    }
};

// 栈
class Solution {
public:
    int calculate(string s) {
        vector<int> stk;
        char preSign = '+';
        int num = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            if (isdigit(s[i])) {
                num = num * 10 + int(s[i] - '0');
            }
            if (!isdigit(s[i]) && s[i] != ' ' || i == n - 1) {
                switch (preSign) {
                    case '+':
                        stk.push_back(num);
                        break;
                    case '-':
                        stk.push_back(-num);
                        break;
                    case '*':
                        stk.back() *= num;
                        break;
                    default:
                        stk.back() /= num;
                }
                preSign = s[i];
                num = 0;
            }
        }
        return accumulate(stk.begin(), stk.end(), 0);
    }
};
```

### [331. 验证二叉树的前序序列化](https://leetcode-cn.com/problems/verify-preorder-serialization-of-a-binary-tree/)

难度：中等 # 2021.03.12

序列化二叉树的一种方法是使用前序遍历。当我们遇到一个非空节点时，我们可以记录下这个节点的值。如果它是一个空节点，我们可以使用一个标记值记录，例如 `#`。

```
     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #
```

例如，上面的二叉树可以被序列化为字符串 `"9,3,4,#,#,1,#,#,2,#,6,#,#"`，其中 `#` 代表一个空节点。

给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。

每个以逗号分隔的字符或为一个整数或为一个表示 `null` 指针的 `'#'` 。

你可以认为输入格式总是有效的，例如它永远不会包含两个连续的逗号，比如 `"1,,3"` 。

**示例 1：**
```
输入: "9,3,4,#,#,1,#,#,2,#,6,#,#"
输出: true
```
**示例 2：**
```
输入: "1,#"
输出: false
```
**示例 3：**
```
输入: "9,#,#,1"
输出: false
```
```cpp
class Solution {
public:
    bool isValidSerialization(string preorder) {
        int len = preorder.size();
        int slots = 1; // 维护剩余槽位数量
        int i = 0;
        while(i < len) {
            if(slots == 0){
                return false;
            }
            if(preorder[i] == ','){
                i++;
            } else if(preorder[i] == '#'){
                slots--;
                i++;
            } else {
                while(i < len && preorder[i] != ','){ // 读一个数字
                    i++;
                }
                slots++; // slots = slots - 1 + 2
            }
        }
        return slots == 0;
    }
};
```

### [150. 逆波兰表达式求值](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

难度：中等 # 2021.03.20

根据 逆波兰表示法，求表达式的值。

有效的算符包括 `+`、`-`、`*`、`/`。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

**说明：**

+ 整数除法只保留整数部分。
+ 给定逆波兰表达式总是有效的。换句话说，表达式总会得出有效数值且不存在除数为 0 的情况。

**示例 1：**
```
输入：tokens = ["2","1","+","3","*"]
输出：9
解释：该算式转化为常见的中缀算术表达式为：((2 + 1) * 3) = 9
```
**示例 2：**

```
输入：tokens = ["4","13","5","/","+"]
输出：6
解释：该算式转化为常见的中缀算术表达式为：(4 + (13 / 5)) = 6
```
**示例 3：**
```
输入：tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
输出：22
解释：
该算式转化为常见的中缀算术表达式为：
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
```

**提示：**

+ `1 <= tokens.length <= 10^4`
+ `tokens[i]` 要么是一个算符（`"+"`、`"-"`、`"*"` 或 `"/"`），要么是一个在范围 `[-200, 200]` 内的整数

**逆波兰表达式：**

逆波兰表达式是一种后缀表达式，所谓后缀就是指算符写在后面。

+ 平常使用的算式则是一种中缀表达式，如 `( 1 + 2 ) * ( 3 + 4 )`。
+ 该算式的逆波兰表达式写法为 `( ( 1 2 + ) ( 3 4 + ) * )`。
逆波兰表达式主要有以下两个优点：

+ 去掉括号后表达式无歧义，上式即便写成 `1 2 + 3 4 + *` 也可以依据次序计算出正确结果。
+ 适合用栈操作运算：遇到数字则入栈；遇到算符则取出栈顶两个数字进行计算，并将结果压入栈中。

```cpp
class Solution {
public:
    int evalRPN(vector<string>& tokens) {
        stack<string> s;
        int len = tokens.size();
        for(int i=0; i<len; i++){
            if(tokens[i] == "+"){
                int b = stoi(s.top()); s.pop();
                int a = stoi(s.top()); s.pop();
                s.push(to_string(a+b));
            } else if(tokens[i] == "-"){
                int b = stoi(s.top()); s.pop();
                int a = stoi(s.top()); s.pop();
                s.push(to_string(a-b));
            } else if(tokens[i] == "*"){
                int b = stoi(s.top()); s.pop();
                int a = stoi(s.top()); s.pop();
                s.push(to_string(a*b));
            } else if(tokens[i] == "/"){
                int b = stoi(s.top()); s.pop();
                int a = stoi(s.top()); s.pop();
                s.push(to_string(a/b));
            } else {
                s.push(tokens[i]);
            }
        }
        return stoi(s.top());
    }
};
```



## 队列

### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

难度：困难 # 2020.01.02

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

**示例 1：**
```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```
**示例 2：**
```
输入：nums = [1], k = 1
输出：[1]
```
**示例 3：**
```
输入：nums = [1,-1], k = 1
输出：[1,-1]
```
**示例 4：**
```
输入：nums = [9,11], k = 2
输出：[11]
```
**示例 5：**
```
输入：nums = [4,-2], k = 2
输出：[4]
```

**提示：**

+ `1 <= nums.length <= 105`
+ `-104 <= nums[i] <= 104`
+ `1 <= k <= nums.length`

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) { // 大根堆
        int len = nums.size();
        priority_queue<pair<int, int>> q; // 优先队列默认大根堆，且pair默认对first排序
        for(int i=0; i<k; i++){
            q.emplace(nums[i], i);
        }
        vector<int> ans = {q.top().first};
        for(int i=k; i<len; i++){
            q.emplace(nums[i], i);
            while(q.top().second <= i - k){
                q.pop();
            }
            ans.push_back(q.top().first);
        }
        return ans;
    }
};

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) { // 单调队列
        int len = nums.size();
        deque<int> q; // 双端队列存索引，索引对应值单调减，后面推出不可能为滑窗最大值的索引，前面推出不在滑窗的索引
        for(int i=0; i<k; i++){
            while(!q.empty() && nums[i] >= nums[q.back()]){
                q.pop_back();
            }
            q.push_back(i);
        }
        vector<int> ans = {nums[q.front()]};
        for(int i=k; i<len; i++){
            while(!q.empty() && nums[i] >= nums[q.back()]){
                q.pop_back();
            }
            q.push_back(i);
            while(q.front() <= i - k){
                q.pop_front();
            }
            ans.push_back(nums[q.front()]);
        }
        return ans;
    }
};
```

### [703. 数据流中的第 K 大元素](https://leetcode-cn.com/problems/kth-largest-element-in-a-stream/)

难度：简单 # 2021.02.12

设计一个找到数据流中第 `k` 大元素的类（class）。注意是排序后的第 `k` 大元素，不是第 `k` 个不同的元素。

请实现 `KthLargest` 类：

+ `KthLargest(int k, int[] nums)` 使用整数 `k` 和整数流 `nums` 初始化对象。
+ `int add(int val)` 将 `val` 插入数据流 `nums` 后，返回当前数据流中第 `k` 大的元素。

**示例：**
```
输入：
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
输出：
[null, 4, 5, 5, 8, 8]

解释：
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8
```

**提示：**
+ `1 <= k <= 10^4`
+ `0 <= nums.length <= 10^4`
+ `-10^4 <= nums[i] <= 10^4`
+ `-10^4 <= val <= 10^4`
+ 最多调用 `add` 方法 `10^4` 次
+ 题目数据保证，在查找第 `k` 大元素时，数组中至少有 `k` 个元素

```cpp
class KthLargest {
public:
    priority_queue<int, vector<int>, greater<int>> q;
    int k;
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (auto& x: nums) {
            add(x);
        }
    }
    
    int add(int val) {
        q.push(val);
        if (q.size() > k) {
            q.pop();
        }
        return q.top();
    }
};

/**
 * Your KthLargest object will be instantiated and called as such:
 * KthLargest* obj = new KthLargest(k, nums);
 * int param_1 = obj->add(val);
 */
```

## 位运算

### [190. 颠倒二进制位](https://leetcode-cn.com/problems/reverse-bits/)

难度：简单 # 2021.03.29

颠倒给定的 32 位无符号整数的二进制位。

**提示：**

+ 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
+ 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 **示例 2** 中，输入表示有符号整数 `-3`，输出表示有符号整数 `-1073741825`。

**进阶：**
如果多次调用这个函数，你将如何优化你的算法？

**示例 1：**
```
输入: 00000010100101000001111010011100
输出: 00111001011110000010100101000000
解释: 输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
     因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
```
示例 2：
```
输入：11111111111111111111111111111101
输出：10111111111111111111111111111111
解释：输入的二进制串 11111111111111111111111111111101 表示无符号整数 4294967293，
     因此返回 3221225471 其二进制表示形式为 10111111111111111111111111111111 。
```

**提示：**

+ 输入是一个长度为 `32` 的二进制字符串

```cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t ans = 0;
        for(int i=0; i<32; i++){
            if(n & (1<<i)){
                ans += (1<<(31-i));
            }
        }
        return ans;
    }
};
```

### [1720. 解码异或后的数组](https://leetcode-cn.com/problems/decode-xored-array/)

难度：简单 # 2021.01.10

**未知** 整数数组 `arr` 由 `n` 个非负整数组成。

经编码后变为长度为 `n - 1` 的另一个整数数组 `encoded` ，其中 `encoded[i] = arr[i] XOR arr[i + 1]` 。例如，`arr = [1,0,2,1]` 经编码后得到 `encoded = [1,2,3]` 。

给你编码后的数组 `encoded` 和原数组 `arr` 的第一个元素 `first`（`arr[0]`）。

请解码返回原数组 `arr` 。可以证明答案存在并且是唯一的。

**示例 1：**
```
输入：encoded = [1,2,3], first = 1
输出：[1,0,2,1]
解释：若 arr = [1,0,2,1] ，那么 first = 1 且 encoded = [1 XOR 0, 0 XOR 2, 2 XOR 1] = [1,2,3]
```
**示例 2：**
```
输入：encoded = [6,2,7,3], first = 4
输出：[4,2,0,7,4]
```

**提示：**

+ `2 <= n <= 10^4`
+ `encoded.length == n - 1`
+ `0 <= encoded[i] <= 10^5`
+ `0 <= first <= 10^5`

```cpp
class Solution {
public:
    vector<int> decode(vector<int>& encoded, int first) {
        int len = encoded.size();
        vector<int> ans(len+1);
        ans[0] = first;
        for(int i=0; i<len; i++){
            ans[i+1] = encoded[i] ^ first;
            first = ans[i+1];
        }        
        return ans;
    }
};
```

### [1486. 数组异或操作](https://leetcode-cn.com/problems/xor-operation-in-an-array/)

难度：简单 # 2021.05.07

给你两个整数，`n` 和 `start` 。

数组 `nums` 定义为：`nums[i] = start + 2*i`（下标从 0 开始）且 `n == nums.length` 。

请返回 `nums` 中所有元素按位异或（**XOR**）后得到的结果。

**示例 1：**
```
输入：n = 5, start = 0
输出：8
解释：数组 nums 为 [0, 2, 4, 6, 8]，其中 (0 ^ 2 ^ 4 ^ 6 ^ 8) = 8 。
     "^" 为按位异或 XOR 运算符。
```
**示例 2：**
```
输入：n = 4, start = 3
输出：8
解释：数组 nums 为 [3, 5, 7, 9]，其中 (3 ^ 5 ^ 7 ^ 9) = 8.
```
**示例 3：**
```
输入：n = 1, start = 7
输出：7
```
**示例 4：**
```
输入：n = 10, start = 5
输出：2
```

**提示：**

+ `1 <= n <= 1000`
+ `0 <= start <= 1000`
+ `n == nums.length`

```cpp
class Solution {
public:
    int xorOperation(int n, int start) {
        vector<int> nums(n);
        nums[0] = start;
        int ans = start;
        for(int i=1; i<n; i++){
            nums[i] = nums[i-1] + 2;
            ans ^= nums[i];
        }
        return ans;
    }
};
```

### [137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

难度：中等 # 2021.05.07

给你一个整数数组 `nums` ，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。

**示例 1：**
```
输入：nums = [2,2,3,2]
输出：3
```
**示例 2：**
```
输入：nums = [0,1,0,1,0,1,99]
输出：99
```

**提示：**

+ `1 <= nums.length <= 3 * 10^4`
+ `-2^31 <= nums[i] <= 2^31 - 1`
+ `nums` 中，除某个元素仅出现 **一次** 外，其余每个元素都恰出现 **三次**

**进阶：**你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) { // 每一位的和除以3的余数
        int ans = 0;
        for(int i=0; i<32; i++){
            int total = 0;
            for(int num: nums){
                total += ((num >> i) & 1);
            }
            if(total % 3){
                ans |= (1 << i);
            }
        }
        return ans;
    }
};
```

## 其他

#### [1603. 设计停车系统](https://leetcode-cn.com/problems/design-parking-system/)

难度：简单 # 2021.03.19

请你给一个停车场设计一个停车系统。停车场总共有三种不同大小的车位：大，中和小，每种尺寸分别有固定数目的车位。

请你实现 `ParkingSystem` 类：

+ `ParkingSystem(int big, int medium, int small)` 初始化 `ParkingSystem` 类，三个参数分别对应每种停车位的数目。
+ `bool addCar(int carType)` 检查是否有 `carType` 对应的停车位。 `carType` 有三种类型：大，中，小，分别用数字 `1`， `2` 和 `3` 表示。一辆车只能停在  `carType` 对应尺寸的停车位中。如果没有空车位，请返回 `false` ，否则将该车停入车位并返回 `true`。

**示例 1：**
```
输入：
["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
[[1, 1, 0], [1], [2], [3], [1]]
输出：
[null, true, true, false, false]

解释：
ParkingSystem parkingSystem = new ParkingSystem(1, 1, 0);
parkingSystem.addCar(1); // 返回 true ，因为有 1 个空的大车位
parkingSystem.addCar(2); // 返回 true ，因为有 1 个空的中车位
parkingSystem.addCar(3); // 返回 false ，因为没有空的小车位
parkingSystem.addCar(1); // 返回 false ，因为没有空的大车位，唯一一个大车位已经被占据了
```

**提示：**

+ `0 <= big, medium, small <= 1000`
+ `carType` 取值为 `1`， `2` 或 `3`
+ 最多会调用 `addCar` 函数 `1000` 次

```cpp
class ParkingSystem {
public:
    int b, m, s;
    ParkingSystem(int big, int medium, int small) {
        b = big;
        m = medium;
        s = small;
    }
    
    bool addCar(int carType) {
        if(carType == 1 && (--b) < 0) return false;
        else if(carType == 2 && (--m) < 0) return false;
        else if(carType == 3 && (--s) < 0) return false;
        return true;
    }
};

/**
 * Your ParkingSystem object will be instantiated and called as such:
 * ParkingSystem* obj = new ParkingSystem(big, medium, small);
 * bool param_1 = obj->addCar(carType);
 */
```



# 剑指 Offer

### [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

难度：简单 # 2020.12.26

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**示例 1：**
```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```
**限制：**

`2 <= n <= 100000`

```cpp
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        int len = nums.size();
        unordered_map<int, int> cnt;
        for(int i=0; i<len; i++){
            cnt[nums[i]]++;
            if(cnt[nums[i]] > 1) return nums[i];
        }
        return -1;
    }
};
```

### [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

难度：中等 # 2020.12.28

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**示例：**

现有矩阵 matrix 如下：

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```
给定 target = `5`，返回 `true`。

给定 target = `20`，返回 `false`。

**限制：**

`0 <= n <= 1000`

`0 <= m <= 1000`

**注意：**本题与主站 240 题相同：https://leetcode-cn.com/problems/search-a-2d-matrix-ii/

```cpp
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int r = matrix.size();
        if(r == 0) return false;
        int c = matrix[0].size();
        if(c == 0) return false;
        int i = r - 1, j = 0;
        while(i >= 0 && i < r && j >= 0 && j < c){
            if(matrix[i][j] == target){
                return true;
            } else if(matrix[i][j] > target) {
                i--;
            } else {
                j++;
            }
        }
        return false;
    }
};
```

### [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

难度：简单 # 2020.12.29

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

**示例 1：**
```
输入：s = "We are happy."
输出："We%20are%20happy."
```

**限制：**

`0 <= s 的长度 <= 10000`

```cpp
class Solution {
public:
    string replaceSpace(string s) {
        int len = s.size();
        string ans;
        for(int i=0; i<len; i++){
            if(s[i] == ' ') ans += "%20";
            else ans += s[i];
        }
        return ans;
    }
};
```

### [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

难度：简单 # 2020.12.30

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

**示例 1：**
```
输入：head = [1,3,2]
输出：[2,3,1]
```
**限制：**

`0 <= 链表长度 <= 10000`

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        vector<int> ans;
        for(ListNode* p = head; p != nullptr; p = p->next){
            ans.push_back(p->val);
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
```

### [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

难度：中等 # 2020.12.31

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如，给出
```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
```
返回如下的二叉树：
```
    3
   / \
  9  20
    /  \
   15   7
```

**限制：**

`0 <= 节点个数 <= 5000`

**注意：**本题与主站 105 题重复：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution { // 递归
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int len = preorder.size();
        if(len == 0) return nullptr;
        int x = preorder[0];
        int idx;
        for(idx=0; idx<len; idx++){
            if(inorder[idx] == x) break;
        }
        vector<int> preorder_l, preorder_r, inorder_l, inorder_r;
        for(int i=0; i<idx; i++){
            preorder_l.push_back(preorder[i+1]);
            inorder_l.push_back(inorder[i]);
        }
        for(int i=idx+1; i<len; i++){
            preorder_r.push_back(preorder[i]);
            inorder_r.push_back(inorder[i]);
        }
        TreeNode *root = new TreeNode(x);
        root->left = buildTree(preorder_l, inorder_l);
        root->right = buildTree(preorder_r, inorder_r);
        return root;
    }
};
```

### [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

难度：简单 # 2021.01.01

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 `appendTail` 和 `deleteHead` ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，`deleteHead` 操作返回 -1 )

**示例 1：**
```
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
```
**示例 2：**
```
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```
**提示：**

+ `1 <= values <= 10000`
+ `最多会对 appendTail、deleteHead 进行 10000 次调用`

```cpp
class CQueue {
public:
    stack<int> stool;
    stack<int> s;

    CQueue() {
        
    }
    
    void appendTail(int value) {
        s.push(value);
    }
    
    int deleteHead() {
        if(s.empty()) return -1;
        while(!s.empty()){
            int t = s.top(); s.pop();
            stool.push(t);
        }
        int ans;
        if(!stool.empty()){
            ans = stool.top(); stool.pop();
        }
        while(!stool.empty()){
            int t = stool.top(); stool.pop();
            s.push(t);
        }
        return ans;
    }
};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */
```

### [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

难度：简单 # 2020.01.03

写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项。斐波那契数列的定义如下：
```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：1
```
**示例 2：**

```
输入：n = 5
输出：5
```

**提示：**

+ `0 <= n <= 100`

```cpp
class Solution {
public:
    int fib(int n) {
        int arr[2] = {0, 1};
        for(int i=2; i<=n; i++){
            arr[i & 1] = (arr[0] + arr[1]) % (int)(1e9 + 7);
        }
        return arr[n & 1];
    }
};
```

### [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

难度：简单 # 2020.01.04

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 `n` 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：2
```
**示例 2：**
```
输入：n = 7
输出：21
```
**示例 3：**
```
输入：n = 0
输出：1
```
**提示：**

+ `0 <= n <= 100`

注意：本题与主站 70 题相同：https://leetcode-cn.com/problems/climbing-stairs/

```cpp
class Solution {
public:
    int numWays(int n) {
        if(n==0)
            return 1;
        if(n==1)
            return 1;
        if(n==2)
            return 2;
        vector<int> dp(n+1,0);
        dp[1] = 1;
        dp[2] = 2;
        for(int i=3; i<=n;i++){
            dp[i] = (dp[i-1] + dp[i-2]) % (int)(1e9 + 7);
        }
        return dp[n];
    }
};
```

### [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

难度：简单 # 2020.01.05

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 `[3,4,5,1,2]` 为 `[1,2,3,4,5]` 的一个旋转，该数组的最小值为1。  

**示例 1：**
```
输入：[3,4,5,1,2]
输出：1
```
**示例 2：**
```
输入：[2,2,2,0,1]
输出：0
```
注意：本题与主站 154 题相同：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/

```cpp
class Solution { // 线性 O(N)
public:
    int minArray(vector<int>& numbers) {
        int len = numbers.size();
        for(int i=1; i<len; i++){
            if(numbers[i-1] > numbers[i]) return numbers[i];
        }
        return numbers[0];
    }
};

class Solution { // 二分 O(logN)
public:
    int minArray(vector<int>& numbers) {
        int len = numbers.size();
        int l = 0, r = len-1;
        if(numbers[l] < numbers[r]) return numbers[l];
        while(l + 1 < r){
            int mid = l + (r-l)/2;
            if(numbers[mid] > numbers[r]) l = mid;
            else if(numbers[mid] < numbers[r]) r = mid;
            else r--;
        }
        return numbers[r];
    }
};
```

### [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)

难度：中等 # 2021.01.07

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的 `3×4` 的矩阵中包含一条字符串 `"bfce"` 的路径（路径中的字母用加粗标出）。

[["a","**b**","c","e"],
["s","**f**","c","s"],
["a","d","**e**","e"]]

但矩阵中不包含字符串 `"abfb"` 的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

示例 1：
```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```
示例 2：
```
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

**提示：**

+ `1 <= board.length <= 200`
+ `1 <= board[i].length <= 200`

注意：本题与主站 79 题相同：https://leetcode-cn.com/problems/word-search/

```cpp
class Solution {
public:
    bool dfs(vector<vector<char>>& board, int r, int c, vector<vector<int>>& v, string& word, int next, int px, int py, int len){
        if(next == len) return true;
        int dx[4] = {0, 1, 0, -1};
        int dy[4] = {1, 0, -1, 0};
        for(int i=0; i<4; i++){
            int x = px + dx[i];
            int y = py + dy[i];
            if(x >= 0 && x < r && y >= 0 && y < c && v[x][y] == 0 && board[x][y] == word[next]){
                v[x][y] = 1;
                if(dfs(board, r, c, v, word, next+1, x, y, len))
                    return true;
                v[x][y] = 0;
            }
        }
        return false;
    }
    bool exist(vector<vector<char>>& board, string word) {
        int r = board.size();
        int c = board[0].size();
        int len = word.size();
        vector<vector<int>> v(r, vector<int>(c, 0));
        for(int i=0; i<r; i++){
            for(int j=0; j<c; j++){
                if(v[i][j] == 0 && board[i][j] == word[0]){
                    v[i][j] = 1;
                    if(dfs(board, r, c, v, word, 1, i, j, len))
                        return true;
                    v[i][j] = 0;
                }
            }
        }
        return false;
    }
};
```

### [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

难度：中等 # 2021.01.08

地上有一个m行n列的方格，从坐标 `[0,0]` 到坐标 `[m-1,n-1]` 。一个机器人从坐标 `[0, 0]` 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

**示例 1：**
```
输入：m = 2, n = 3, k = 1
输出：3
```
**示例 2：**
```
输入：m = 3, n = 1, k = 0
输出：1
```
**提示：**

+ `1 <= n,m <= 100`
+ `0 <= k <= 20`

```cpp
class Solution {
public:
    bool islegal(int x, int y, int k){
        int cnt = 0;
        string s = to_string(x);
        int len = s.size();
        for(int i=0; i<len; i++){
            cnt += s[i] - '0';
        }
        s = to_string(y);
        len = s.size();
        for(int i=0; i<len; i++){
            cnt += s[i] - '0';
        }
        return cnt <= k;
    }
    void dfs(vector<vector<int>>& v, int m, int n, int x, int y, int k, int& cnt){
        v[x][y] = 1;
        if(islegal(x, y, k)){
            cnt++;
            int dx[4] = {0, 1, 0, -1};
            int dy[4] = {-1, 0, 1, 0};
            for(int i=0; i<4; i++){
                int xx = x + dx[i];
                int yy = y + dy[i];
                if(xx >= 0 && xx < m && yy >= 0 && yy < n && v[xx][yy] == 0){
                    dfs(v, m, n, xx, yy, k, cnt);
                }
            }
        }
    }
    int movingCount(int m, int n, int k) {
        vector<vector<int>> v(m, vector<int>(n, 0));
        int cnt = 0;
        dfs(v, m, n, 0, 0, k, cnt);
        return cnt;
    }
};
```

### [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

难度：中等 # 2021.01.08

给你一根长度为 `n` 的绳子，请把绳子剪成整数长度的 `m` 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m-1]` 。请问 `k[0]*k[1]*...*k[m-1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

**示例 1：**
```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```
**示例 2：**
```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```
**提示：**

+ `2 <= n <= 58`

注意：本题与主站 343 题相同：https://leetcode-cn.com/problems/integer-break/

```cpp
class Solution {
public:
    int cuttingRope(int n) {
        vector<int> dp(n+1, 0);
        dp[1] = 1;
        dp[2] = 1;
        for(int i=3; i<=n; i++){
            for(int j=1; j<=i/2; j++){
                dp[i] = max(dp[i], max(dp[j], j) * max(dp[i-j], i-j)); // 如果自己单独作为一段比分段长，比如3单独分一段为3，但分段后为2
            }
        }
        return dp[n];
    }
};
```

### [剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

难度：中等 # 2021.01.08

给你一根长度为 `n` 的绳子，请把绳子剪成整数长度的 `m` 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m - 1]` 。请问 `k[0]*k[1]*...*k[m - 1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**
```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```
**示例 2：**
```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```
**提示：**

+ `2 <= n <= 1000`

注意：本题与主站 343 题相同：https://leetcode-cn.com/problems/integer-break/

```cpp
// 任何大于1的数都可由2和3相加组成
// 当n>=5时,将它剪成2或3的绳子段,2(n-2) > n,3(n-3) > n,都大于他未拆分前的情况，
// 当n>=5时，3(n-3) >= 2(n-2),所以我们尽可能地多剪3的绳子段
// 当绳子长度被剪到只剩4时，2 * 2 = 4 > 1 * 3,所以没必要继续剪

class Solution {
public:
    int cuttingRope(int n) {
        if(n <= 3) return n-1; // 如果n<=3,数字要求至少分为两部分，实际结果的最大值为n-1
        long long ans = 1;
        while(n > 4)
        {
            n = n - 3;
            ans = ans * 3 % 1000000007;
        }
        return ans * n % 1000000007;
    }
};
```

### [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

难度：简单 # 2021.01.08

请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

**示例 1：**
```
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```
**示例 2：**
```
输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```
**示例 3：**
```
输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
```

**提示：**

+ 输入必须是长度为 32 的 **二进制串** 。

注意：本题与主站 191 题相同：https://leetcode-cn.com/problems/number-of-1-bits/

```cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        int cnt = 0;
        while(n){
            if(n & 1) cnt++;
            n >>= 1;
        }
        return cnt;
    }
};
```

### [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

难度：中等 # 2021.01.09

实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。

**示例 1：**
```
输入: 2.00000, 10
输出: 1024.00000
```
**示例 2：**
```
输入: 2.10000, 3
输出: 9.26100
```
**示例 3：**
```
输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```
**说明：**

+ -100.0 < x < 100.0
+ n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

注意：本题与主站 50 题相同：https://leetcode-cn.com/problems/powx-n/

```cpp
class Solution {
public:
    double myPow(double x, int n) {
        if(n == 0) return 1.0;
        if(n == 1) return x;
        if(n < 0) return 1.0/x / myPow(x, -(n+1)); // 为了过 1.00000,-2147483648 测试用例
        return (n%2==0) ? myPow(x*x, n>>1) : x * myPow(x*x, (n-1)>>1); // 能过 0.00001,2147483647 测试用例
    }
};

class Solution {
public:
    double myPow(double x, long long n) {
        if(n == 0)
            return 1;
        else if(n < 0)
            return 1 / myPow(x, -n);
        else{
            if(n % 2 == 0){
                double half = myPow(x, n/2);
                return half * half;
            }else{
                return myPow(x, n-1) * x;
            }
        }
    }
};
```



### [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

难度：中等 # 2021.01.18

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串 `"+100"`、`"5e2"`、`"-123"`、`"3.1416"`、`"-1E-16"`、`"0123"` 都表示数值，但 `"12e"`、`"1a3.14"`、`"1.2.3"`、`"+-5"` 及 `"12e+5.4"` 都不是。

```cpp
class Solution { // 正则表达式（超时）
public:
    bool isNumber(string s) {
        s.erase(0,s.find_first_not_of(" ")); 
        s.erase(s.find_last_not_of(" ") + 1);
        regex reg0("\\s*|([+-]?\\.?[eE][\\s\\S]*)|([+-]?\\.)"); // 缺数字的情况
        if(regex_match(s, reg0)) return false;
        regex reg2("([\ ]*)(([+-])?\\d*\\.?\\d*)([eE][+-]?\\d+)?([\ ]*)([\ ]*)");
        return regex_match(s, reg2);
    }
};

class Solution { // 将字符串划分为3部分，A:小数点之前 B:小数点之后,e或E之前 C:e或E之后。A和B只需存在一个，若e或E存在，C必须存在。
public:
    bool scanNumber(string::iterator& it, string::iterator& s_end){//找数字，找不到返回false
        bool have_number = false;
        while(it!=s_end && (*it)>='0' && (*it)<='9'){
            have_number=true;
            it++;
        }
        return have_number;
    }

    bool scanInteger(string::iterator& it, string::iterator& s_end){//找数字，找不到返回false(首位可为+或-)
        if(*it=='+' || *it=='-') it++;
        return scanNumber(it, s_end);
    }
    
    bool isNumber(string s) {
        if(s.empty()) return false;
        auto it = s.begin();
        auto s_end = s.end();
        while(*it == ' ') it++; // 去空格
        bool ans = scanInteger(it, s_end); // 找A整数位
        if(*it == '.'){ // 存在小数点，则找B小数位
            it++;
            ans |= scanNumber(it, s_end); // 小数位和整数位存在一个即可
        }
        if(*it=='e' || *it=='E'){ // 存在e或E，则找指数位
            it++;
            ans &= scanInteger(it, s_end);//若有e或E，指数位C必须存在
            
        }     
        while(*it == ' ') it++; // 去空格
        return ans && it==s.end(); // 若到达结尾且满足之前规则，返回true
    }
};
```

### [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

难度：简单 # 2020.01.18

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

**示例：**

```
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
```

**提示：**

1. `1 <= nums.length <= 50000`
2. `1 <= nums[i] <= 10000`

```cpp
class Solution { // 线性
public:
    vector<int> exchange(vector<int>& nums) {
        vector<int> ans, even;
        int len = nums.size();
        for(int i=0; i<len; i++){
            if(nums[i]%2 == 1) ans.push_back(nums[i]);
            else even.push_back(nums[i]);
        }
        ans.insert(ans.end(), even.begin(), even.end());
        return ans;
    }
};

class Solution { // 双指针
public:
    vector<int> exchange(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            if ((nums[left] & 1) == 1) { // 左侧开始找奇数，偶数时停下
                left++;
                continue;
            }
            if ((nums[right] & 1) == 0) { // 右侧开始找偶数，奇数时停下
                right--;
                continue;
            }
            swap(nums[left++], nums[right--]); // 交换
        }
        return nums;
    }
};
```

### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

难度：简单 # 2020.01.18

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
```

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution { // 需要遍历一次获取长度
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        int len=0, cnt=0;
        for(ListNode* p = head; p!=nullptr; p=p->next){
            len++;
        }
        ListNode* ans = head;
        for(; ans!=nullptr; ans=ans->next){
            cnt++;
            if(cnt == len-k+1)
                break;
        }
        return ans;
    }
};

class Solution { // 前面的先走k步看看
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        int cnt=0;
        ListNode *fast=head, *slow=head;
        for(; fast!=nullptr; fast=fast->next){
            cnt++;
            if(cnt == k+1) break;
        }
        while(fast!=nullptr){
            cnt++;
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }
};
```

### [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

难度：简单 # 2020.01.18

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

**示例:**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

**限制：**

```
0 <= 节点个数 <= 5000
```

**注意**：本题与主站 206 题相同：https://leetcode-cn.com/problems/reverse-linked-list/

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) { // 逐个翻转
        ListNode* rear = NULL;
        ListNode* p = head;
        while(p != NULL){
            ListNode* nextnode = p->next;
            p->next = rear;
            rear = p;
            p = nextnode;
        }
        return rear;
    }
};

class Solution { // 递归
public:
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* ans = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return ans;
    }
};
```

### [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

难度：简单 # 2021.01.20

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

**示例1：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

**限制：**

`0 <= 链表长度 <= 1000`

注意：本题与主站 21 题相同：https://leetcode-cn.com/problems/merge-two-sorted-lists/

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode* head = new ListNode(0);
        ListNode* p = head;
        while(l1!=nullptr && l2!=nullptr){
            if(l1->val < l2->val){
                p->next = l1; p=p->next; l1=l1->next;
            } else {
                p->next = l2; p=p->next; l2=l2->next;
            }
        }
        if(l1 != nullptr) p->next = l1;
        if(l2 != nullptr) p->next = l2;
        return head->next;
    }
};
```

### [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

难度：中等 # 2021.01.21

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:
```
   3
  / \
  4  5
 / \
 1  2
```


给定的树 B：
```
   4 
  /
 1
```
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

**示例 1：**

```
输入：A = [1,2,3], B = [3,1]
输出：false
```
**示例 2：**

```
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```
限制：

`0 <= 节点个数 <= 10000`

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool isEqual(TreeNode* A, TreeNode* B){
        if(B == nullptr) return true;
        if(A == nullptr) return false;
        if(A->val != B->val) return false;
        return isEqual(A->left, B->left) && isEqual(A->right, B->right);
    }
    bool isSub(TreeNode* A, TreeNode* B) {
        if(B == nullptr) return true;
        if(A == nullptr) return false;
        if(A->val == B->val){
            if(isEqual(A, B)) return true;
        }
        return isSub(A->left, B) || isSub(A->right, B);
    }
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if(B == nullptr) return false;
        return isSub(A, B);
    }
};
```

### [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

难度 ：简单 # 2021.01.23

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的：
```
    1
   / \
  2   2
   \   \
    3   3
```

**示例 1：**
```
输入：root = [1,2,2,3,4,4,3]
输出：true
```
**示例 2：**
```
输入：root = [1,2,2,null,3,null,3]
输出：false
```
**限制：**

`0 <= 节点个数 <= 1000`

注意：本题与主站 101 题相同：https://leetcode-cn.com/problems/symmetric-tree/

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool help(TreeNode* l , TreeNode* r) {
        if((l==nullptr) ^ (r==nullptr)) return false;
        if(l==nullptr && r==nullptr) return true;
        return (l->val == r->val) && help(l->left, r->right) && help(l->right, r->left);
    }
    bool isSymmetric(TreeNode* root) {
        return root == nullptr ? true : help(root->left , root->right);
    }
};
```

### [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

难度：简单 # 2021.01.24

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

**示例 1：**
```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```
**示例 2：**
```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

**限制：**

+ `0 <= matrix.length <= 100`
+ `0 <= matrix[i].length <= 100`

注意：本题与主站 54 题相同：https://leetcode-cn.com/problems/spiral-matrix/

```cpp
class Solution {
public:
    void circle(vector<vector<int>>& matrix, int a, int b, int c, int d, vector<int>& ans){
        ans.push_back(matrix[a][b]);
        if(b+1 <= d){ // 有第一段
            for(int y=b+1; y<=d; y++) ans.push_back(matrix[a][y]);
        }
        if(a+1 <= c){ // 有第二段
            for(int x=a+1; x<=c; x++) ans.push_back(matrix[x][d]);
        }
        if(b+1 <= d && a+1 <= c){ // 有第三段
            for(int y=d-1; y>=a; y--) ans.push_back(matrix[c][y]);
        }
        if(b+1 <= d && a+1 <= c){ // 第四段
            for(int x=c-1; x>a; x--) ans.push_back(matrix[x][b]);
        }
    }
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int r = matrix.size();
        int c = matrix[0].size();
        vector<int> ans;
        for(int i=0; i<=min((r-1)/2, (c-1)/2); i++){
            circle(matrix, i, i, r-1-i, c-1-i, ans);
        }
        return ans;
    }
};
```

### [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

难度：简单 # 远古

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

**示例：**
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

**提示：**

1. 各函数的调用总次数不超过 20000 次

注意：本题与主站 155 题相同：https://leetcode-cn.com/problems/min-stack/

```cpp
class MinStack {
public:
    /** initialize your data structure here. */
    vector<int> s;
    MinStack() {

    }
    
    void push(int x) {
        this->s.push_back(x);
    }
    
    void pop() {
        this->s.pop_back();
    }
    
    int top() {
        return this->s[this->s.size()-1];
    }
    
    int min() {
        return this->s[min_element(this->s.begin(), this->s.end())-this->s.begin()];
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
```

### [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

难度：中等 # 2021.01.24

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

**示例 1：**
```
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```
**示例 2：**
```
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```
**提示：**

1. `0 <= pushed.length == popped.length <= 1000`
2. `0 <= pushed[i], popped[i] < 1000`
3. `pushed` 是 `popped` 的排列。

注意：本题与主站 946 题相同：https://leetcode-cn.com/problems/validate-stack-sequences/

```cpp
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) { // 模拟
        int len = pushed.size();
        if(len == 0) return true;
        vector<int> v;
        int p;
        for(p=0; p<len; p++){
            if(pushed[p] != popped[0]) v.push_back(pushed[p]);
            else break;
        }
        for(int i=1; i<len; i++){
            if(find(v.begin(), v.end(), popped[i]) != v.end()){ // 在栈内
                if(popped[i] != v[v.size()-1]){ // 不是栈尾元素
                    return false;
                } else {
                    v.pop_back();
                }
            } else { // 不在栈内，就把该元素在其入栈顺序前的所有元素加入栈中
                for(p++; p<len; p++){
                    if(pushed[p] != popped[i]) v.push_back(pushed[p]);
                    else break;
                }
            }
        }
        if(v.size() == 0) return true;
        return false;
    }
};
```

