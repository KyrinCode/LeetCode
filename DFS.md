### 2698. Find the Punishment Number of an Integer

https://leetcode.cn/problems/find-the-punishment-number-of-an-integer/description/

Medium # 2023/10/25

```go
func punishmentNumber(n int) int {
    res := 0
    for i := 0; i <= n; i++ {
        if dfs(strconv.Itoa(i * i), 0, 0, i) {
            res += i * i
        }
    }
    return res
}

func dfs(s string, pos, total, target int) bool {
    if pos == len(s) {
        return total == target
    }
    sum := 0
    for i := pos; i < len(s); i++ {
        sum = sum * 10 + int(s[i] - '0')
        if sum + total > target {
            break
        }
        if dfs(s, i+1, sum+total, target) {
            return true
        }
    }
    return false
}
```

### 1240. Tiling a Rectangle with the Fewest Squares

https://leetcode.cn/problems/tiling-a-rectangle-with-the-fewest-squares/

Hard # 2023/10/26

```	go
func tilingRectangle(n int, m int) int {
    grid := make([][]int, n)
    for i, _ := range grid {
        grid[i] = make([]int, m)
    }
    // grid := [n][m]int{}
    ans := n * m
    dfs(&grid, &ans, 0, 0, 0)
    return ans
}

func fill(grid *[][]int, x, y, l, val int) {
    for i := 0; i < l; i++ {
        for j := 0; j < l; j++ {
            (*grid)[x+i][y+j] += val
        }
    }
}

func dfs(grid *[][]int, ans *int, x, y, cnt int) {
    n := len(*grid)
    m := len((*grid)[0])
    if cnt >= *ans { // 剪枝
        return
    }
    if x == n {
        *ans = cnt
        return
    }
    if (*grid)[x][y] == 1 { // 当前位置上过色
        j := y + 1
        for j < m && (*grid)[x][j] == 1 {
            j++
        }
        if j > y && j < m {
            dfs(grid, ans, x, j, cnt) // 右侧如果还有没上色的，从右侧继续
        } else {
            dfs(grid, ans, x+1, 0, cnt) // 如果右侧都上过色了，从下一行行首开始
        }
    } else { // 当前位置未上色
        lmax := 1
        for x + lmax - 1 < n && y + lmax - 1 < m && (*grid)[x][y + lmax - 1] == 0 {
            lmax++
        }
        lmax--
        for l := lmax; l >= 1; l-- { // 从最长边长降序搜索减少分支
            fill(grid, x, y, l, 1)
            if y + l < m { // 右侧又有没上色的
                dfs(grid, ans, x, y+l, cnt+1)
            } else { // 右侧都上过色了
                dfs(grid, ans, x+1, 0, cnt+1)
            }
            fill(grid, x, y, l, -1)
        }
    }
}
```

### 78. Subsets

https://leetcode.cn/problems/subsets/

Medium # 2023/10/26

```go
func subsets(nums []int) [][]int {
    res := [][]int{}
    backtrack(&res, nums, 0, []int{})
    return res
}

func dfs(res *[][]int, nums []int, pos int, list []int) {
    if pos == len(nums) {
        // wrong, should copy, or the address is not changed
        // *res = append(*res, list)
        
        // option 1: copy with copy
        nlist := make([]int, len(list))
        copy(nlist, list)
        *res = append(*res, nlist)

        // option 2: copy with append
        // *res = append(*res, append([]int{}, list...))
        return
    } else {
        dfs(res, nums, pos+1, append(list, nums[pos]))
        dfs(res, nums, pos+1, list)
    }
}
```

### 2003. Smallest Missing Genetic Value in Each Subtree

https://leetcode.cn/problems/smallest-missing-genetic-value-in-each-subtree/

Hard # 2023/10/31

```go
// 答案初始化为1 再从根节点开始对每一个节点dfs

func smallestMissingValueSubtree(parents []int, nums []int) []int {
    children := make([][]int, len(nums))
    for i := 1; i < len(nums); i++ {
        children[parents[i]] = append(children[parents[i]], i)
    }
    res := make([]int, len(nums))
    for i := 0; i < len(nums); i++ {
        res[i] = 1
    }
    var dfs func(int) (map[int]bool, int)
    dfs = func(node int) (map[int]bool, int) {
        geneSet := make(map[int]bool)
        geneSet[nums[node]] = true
        for _, child := range children[node] {
            geneSetChild, resChild := dfs(child)
            // 节点缺少的最小值一定大于等于子节点的
            res[node] = max(res[node], resChild)
            // 节点值集为孩子节点值集的并集
            if len(geneSet) < len(geneSetChild) {
                geneSet, geneSetChild = geneSetChild, geneSet
            }
            for key := range geneSetChild {
                geneSet[key] = true
            }
        }
        // 确定该节点缺少的最小值
        for geneSet[res[node]] {
            res[node]++
        }
        return geneSet, res[node]
    }
    dfs(0)
    return res
}

func max(a, b int) int {
    if a > b {
        return a
    } else {
        return b
    }
}
```

```go
// 答案初始化为1 在从值为1的节点dfs得到值集 并循环其祖先更新值集

func smallestMissingValueSubtree(parents []int, nums []int) []int {
    children := make([][]int, len(nums))
    for i := 1; i < len(nums); i++ {
        children[parents[i]] = append(children[parents[i]], i)
    }
    res := make([]int, len(nums))
    // 找到值为1的节点
    var node1 int
    for i := 0; i < len(nums); i++ {
        res[i] = 1
        if nums[i] == 1 {
            node1 = i
        }
    }
    // dfs遍历某点的子树 更新值集
    geneSet := make(map[int]bool)
    visited := make([]bool, len(nums))
    var dfs func(int)
    dfs = func(node int) {
        if visited[node] {
            return
        }
        visited[node] = true
        geneSet[nums[node]] = true
        for _, child := range children[node] {
            dfs(child)
        }
    }
    // 从值为去的节点向上dfs直至parent为-1
    resIncre := 1
    for node1 != -1 {
        dfs(node1)
        for geneSet[resIncre] {
            resIncre++
        }
        res[node1] = resIncre
        node1 = parents[node1]
    }
    return res
}
```

### 1457. Pseudo-Palindromic Paths in a Binary Tree

https://leetcode.cn/problems/pseudo-palindromic-paths-in-a-binary-tree/

Medium # 2023/11/25

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func pseudoPalindromicPaths (root *TreeNode) int {
    cnt := make([]int, 9)
    res := 0
    var dfs func(*TreeNode)
    dfs = func(root *TreeNode) {
        if root == nil {
            return
        }
        cnt[root.Val-1]++
        if root.Left == nil && root.Right == nil {
            // judge
            odd, i := 0, 0
            for ; i < 9; i++ {
                if cnt[i] % 2 == 1 {
                    odd++
                    if odd > 1 {
                        break
                    }
                }
            }
            if i == 9 {
                res++
            }
        } else {
            dfs(root.Left)
            dfs(root.Right)
        }        
        cnt[root.Val-1]--
    }
    dfs(root)
    return res
}
```

### 395. Longest Substring with At Least K Repeating Characters

https://leetcode.cn/problems/longest-substring-with-at-least-k-repeating-characters/

Medium # 2023/12/01

```go
func longestSubstring(s string, k int) int {
    var dfs func(int, int, int) int
    dfs = func(start, end, k int) int {
        cnt := make(map[byte]int)
        for i := start; i <= end; i++ {
            cnt[s[i]]++
        }
        flag := false
        for _, v := range cnt {
            if v < k {
                flag = true
                break
            }
        }
        if !flag {
            return end - start + 1
        }

        res := 0
        i := start
        for i <= end {
            for i <= end && cnt[s[i]] < k {
                i++
            }
            if i == end + 1 {
                break
            }
            newStart := i
            for i <= end && cnt[s[i]] >= k {
                i++
            }
            newEnd := i - 1
            res = max(res, dfs(newStart, newEnd, k))
        }
        return res
    }
    return dfs(0, len(s)-1, k)
}
```

```go
// 滑动窗口
func longestSubstring(s string, k int) int {
    cnt := make(map[byte]int)
    for i := 0; i < len(s); i++ {
        cnt[s[i]]++
    }
    uplmt := 0
    for _, _ = range cnt {
        uplmt++
    }
    res := 0
    for num := 1; num <= uplmt; num++ { // num: 包含了num个不同字符
        window := make(map[byte]int)
        cCnt := 0 // 不同字符数量
        lessK := 0 // 小于k的字符数量
        left, right := 0, 0 // 左闭右开
        for right < len(s) {
            c := s[right]
            right++
            if window[c] == 0 {
                cCnt++
                lessK++
            }
            window[c]++
            if window[c] == k {
                lessK--
            }
            // update
            if cCnt == num && lessK == 0 {
                res = max(res, right - left)
            }
            for cCnt > num {
                // shrink
                c = s[left]
                left++
                if window[c] == k {
                    lessK++
                }
                window[c]--
                if window[c] == 0 {
                    cCnt--
                    lessK--
                }
            }
        }
    }
    return res
}
```

### 1901. Find a Peak Element II

https://leetcode.cn/problems/find-a-peak-element-ii/

Medium # 2023/12/19

```go
func findPeakGrid(mat [][]int) []int {
    r, c := len(mat), len(mat[0])
    dir := [][]int{
        {0, 1},
        {0, -1},
        {1, 0},
        {-1, 0},
    }
    leftRow, leftCol, rightRow, rightCol := 0, 0, r - 1, c - 1
    midRow, midCol := leftRow + (rightRow - leftRow) / 2, leftCol + (rightCol - leftCol) / 2
    // 往高处搜索
    var search func(int, int) (int, int)
    search = func(startRow, startCol int) (int, int) {
        flag := false
        for k := 0; k < 4; k++ {
            row, col := startRow + dir[k][0], startCol + dir[k][1]
            if row >= 0 && row < r && col >= 0 && col < c && mat[row][col] >= mat[startRow][startCol] {
                flag = true
                return search(row, col)
            }
        }
        if flag == false {
            return startRow, startCol
        }
        return -1, -1 // 不可能
    }
    res0, res1 := search(midRow, midCol)
    return []int{res0, res1}
}
```

### 1690. Stone Game VII

https://leetcode.cn/problems/stone-game-vii/

Medium # 2024/02/03

```go
// 记忆化搜索
func stoneGameVII(stones []int) int {
    l := len(stones)
    sum := make([]int, l+1)
    for i := 0; i < l; i++ {
        sum[i+1] = sum[i] + stones[i]
    }
    memo := make([][]int, l)
    for i := 0; i < l; i++ {
        memo[i] = make([]int, l)
    }

    var dfs func(int, int) int
    dfs = func(i, j int) int { // 左闭右闭区间下可以获得最小差值
        if i >= j {
            return 0
        }
        if memo[i][j] == 0 {
            memo[i][j] = max(sum[j+1] - sum[i+1] - dfs(i+1, j), sum[j] - sum[i] - dfs(i, j-1))
        }   
        return memo[i][j]
    }

    return dfs(0, l-1)
}
```

### 236. Lowest Common Ancestor of a Binary Tree

https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/

Medium # 2024/02/09

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    var res *TreeNode

    var dfs func(*TreeNode) (bool, bool)
    dfs = func(root *TreeNode) (bool, bool) {
        if root == nil {
            return false, false
        }
        leftHasP, leftHasQ := dfs(root.Left)
        rightHasP, rightHasQ := dfs(root.Right)
        hasP := leftHasP || rightHasP
        hasQ := leftHasQ || rightHasQ
        if root.Val == p.Val {
            hasP = true
        } else if root.Val == q.Val {
            hasQ = true
        }
        if hasP && hasQ {
            res = root
        }
        if res != nil {
            return false, false
        }
        return hasP, hasQ
    }
    dfs(root)
    return res
}
```

### 987. Vertical Order Traversal of a Binary Tree

https://leetcode.cn/problems/vertical-order-traversal-of-a-binary-tree/)

Hard # 2024/02/13

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func verticalTraversal(root *TreeNode) [][]int {
    nodes := []Node{}
    var dfs func(*TreeNode, int, int)
    dfs = func(root *TreeNode, row, col int) {
        if root == nil {
            return
        }
        nodes = append(nodes, Node{root.Val, row, col})
        dfs(root.Left, row+1, col-1)
        dfs(root.Right, row+1, col+1)
    }
    dfs(root, 0, 0)

    sort.Slice(nodes, func(i, j int) bool {
        a, b := nodes[i], nodes[j]
        return a.Col < b.Col || a.Col == b.Col && a.Row < b.Row || a.Col == b.Col && a.Row == b.Row && a.Val < b.Val
    })

    res := [][]int{}
    slow, fast := 0, 0
    for fast < len(nodes) {
        tmp := []int{}
        for fast < len(nodes) && nodes[fast].Col == nodes[slow].Col {
            tmp = append(tmp, nodes[fast].Val)
            fast++
        }
        res = append(res, tmp)
        slow = fast
    }
    return res
}

type Node struct {
    Val int
    Row int
    Col int
}
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/

Medium # 2024/02/20

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func buildTree(preorder []int, inorder []int) *TreeNode {
    l := len(preorder)
    if l == 0 {
        return nil
    }
    rootVal := preorder[0]
    pos := 0
    for ; pos < l; pos++ {
        if rootVal == inorder[pos] {
            break
        }
    }
    left := buildTree(preorder[1:1+pos], inorder[:pos])
    right := buildTree(preorder[1+pos:], inorder[1+pos:])
    root := &TreeNode {
        Val: preorder[0],
        Left: left,
        Right: right,
    }
    return root
}
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/

Medium # 2024/02/21

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func buildTree(inorder []int, postorder []int) *TreeNode {
    l := len(inorder)
    if l == 0 {
        return nil
    }
    rootVal := postorder[l-1]
    pos := 0
    for ; pos < l; pos++ {
        if inorder[pos] == rootVal {
            break
        }
    }
    left := buildTree(inorder[:pos], postorder[:pos])
    right := buildTree(inorder[pos+1:], postorder[pos:l-1])
    root := &TreeNode {
        Val: rootVal,
        Left: left,
        Right: right,
    }
    return root
}
```

### 889. Construct Binary Tree from Preorder and Postorder Traversal

https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/

Medium # 2024/02/22

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */

func constructFromPrePost(preorder []int, postorder []int) *TreeNode {
    l := len(preorder)
    if l == 0 {
        return nil
    }
    rootVal := preorder[0]
    if l == 1 {
        return &TreeNode {
            Val: rootVal,
            Left: nil,
            Right: nil,
        }
    }
    preMap, postMap := make(map[int]bool), make(map[int]bool)
    pos := 0
    for ; pos < l-1; pos++ {
        if _, ok := postMap[preorder[1+pos]]; ok {
            delete(postMap, preorder[1+pos])
        } else {
            preMap[preorder[1+pos]] = true
        }
        if _, ok := preMap[postorder[pos]]; ok {
            delete(preMap, postorder[pos])
        } else {
            postMap[postorder[pos]] = true
        }
        if len(preMap) == 0 && len(postMap) == 0 {
            break
        }
    }
    left := constructFromPrePost(preorder[1:pos+2], postorder[:pos+1])
    right := constructFromPrePost(preorder[pos+2:], postorder[pos+1:l-1])
    return &TreeNode {
        Val: rootVal,
        Left: left,
        Right: right,
    }
}
```

### 2673. Make Costs of Paths Equal in a Binary Tree

https://leetcode.cn/problems/make-costs-of-paths-equal-in-a-binary-tree/

Medium # 2024/02/28

```go
// 自底向上，同parent的两个child补齐最大值
func minIncrements(n int, cost []int) int {
    res := 0
    var dfs func(int)
    dfs = func(i int) {
        if 2*i+2 < n {
            dfs(2*i+1)
            dfs(2*i+2)
            cost[i] += max(cost[2*i+1], cost[2*i+2])
            diff := cost[2*i+1] - cost[2*i+2]
            if diff < 0 {
                diff *= -1
            }
            res += diff
        }
    }
    dfs(0)
    return res
}
```

### 2368. Reachable Nodes With Restrictions

https://leetcode.cn/problems/reachable-nodes-with-restrictions/

Medium # 2024/03/02

```go
func reachableNodes(n int, edges [][]int, restricted []int) int {
    isRestricted := make([]bool, n)
    for _, x := range restricted {
        isRestricted[x] = true
    }
    graph := make([][]int, n)
    for _, edge := range edges {
        graph[edge[0]] = append(graph[edge[0]], edge[1])
        graph[edge[1]] = append(graph[edge[1]], edge[0])
    }

    cnt := 0
    visited := make([]bool, n)
    var dfs func(int)
    dfs = func(root int) {
        cnt++
        visited[root] = true
        for _, node := range graph[root] {
            if !visited[node] && !isRestricted[node] {
                dfs(node)
            }
        }
    }
    dfs(0)
    return cnt
}
```

