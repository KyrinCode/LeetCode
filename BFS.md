### 111. Minimum Depth of Binary Tree

https://leetcode.cn/problems/minimum-depth-of-binary-tree/

Simple # 2023/11/01

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func minDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    depth := 1
    q := []*TreeNode{root}
    for len(q) > 0 {
        l := len(q)
        for i := 0; i < l; i++ { // 第二个for是为了方便记录每层 BFS如此得到的一定是最短
            node := q[0]
            q = q[1:]
            if node.Left == nil && node.Right == nil { // 判断target条件
                return depth
            }
            if node.Left != nil {
                q = append(q, node.Left)
            }
            if node.Right != nil {
                q = append(q, node.Right)
            }
        }
        depth++
    }
    return depth
}
```

### 752. Open the Lock

https://leetcode.cn/problems/open-the-lock/

Medium # 2023/11/02

```go
// BFS 最短路径 搜索从起点0000到targetWXYZ的最短路径长度，排除中间不可到达的deadends
// 枚举每层每个状态XXXX下8种下一步中所有合法状态加入队列作为新的一层

func openLock(deadends []string, target string) int {
    visited := make(map[string]bool)
    for _, deadend := range deadends {
        visited[deadend] = true
    }
    if visited["0000"] == true {
        return -1
    }

    depth := 0
    q := []string{"0000"}
    visited["0000"] = true // 加进队列的时候就算visit了 不然会有重复的加进队列 导致q无限增大
    for len(q) > 0 {
        l := len(q)
        for i := 0; i < l; i++ {
            state := q[0]
            q = q[1:]
            if state == target {
                return depth
            }
            for pos := 0; pos < 4; pos++ {
                for _, flag := range []int{1, -1} {
                    nextState := turn(state, pos, flag)
                    if !visited[nextState] {
                        q = append(q, nextState)
                        visited[nextState] = true
                    }
                }
            }
        }
        depth++
    }
    return -1
}

func turn(state string, pos, flag int) string { // flag: +1 -1
    if flag == 1 {
        return state[:pos] + string(((state[pos] - '0') + 1) % 10 + '0') + state[pos+1:]
    } else {
        return state[:pos] + string(((state[pos] - '0') + 9) % 10 + '0') + state[pos+1:]
    }
}
```

### 773. Sliding Puzzle

https://leetcode.cn/problems/sliding-puzzle/

Hard # 2023/11/03

```go
func slidingPuzzle(board [][]int) int {
    start := encode(board)
    visited := make(map[string]bool)
    q := []string{start}
    visited[start] = true
    depth := 0
    for len(q) > 0 {
        l := len(q)
        for i := 0; i < l; i++ {
            state := q[0]
            q = q[1:]
            if state == "123450" {
                return depth
            }
            pos0 := 0
            for ; pos0 < 6; pos0++ {
                if state[pos0] == '0' {
                    break
                }
            }
            var newState string
            if pos0 % 3 != 0 { // 不是第一列
                newState = swap(state, pos0-1, pos0)
                if !visited[newState] {
                    q = append(q, newState)
                    visited[newState] = true
                }
            }
            if pos0 / 3 != 0 { // 不是第一行
                newState = swap(state, pos0-3, pos0)
                if !visited[newState] {
                    q = append(q, newState)
                    visited[newState] = true
                }
            }
            if pos0 % 3 != 2 { // 不是最后一列
                newState = swap(state, pos0, pos0+1)
                if !visited[newState] {
                    q = append(q, newState)
                    visited[newState] = true
                }
            }
            if pos0 / 3 != 1 { // 不是最后一行
                newState = swap(state, pos0, pos0+3)
                if !visited[newState] {
                    q = append(q, newState)
                    visited[newState] = true
                }
            }
        }
        depth++
    }
    return -1
}

func encode(board [][]int) string {
    s := ""
    for _, row := range board {
        for _, cell := range row {
            s += strconv.Itoa(cell)
        }
    }
    return s
}

func swap(state string, i, j int) string { // i在j前
    return state[:i] + string(state[j]) + state[i+1:j] + string(state[i]) + state[j+1:]
}
```

### 2258. Escape the Spreading Fire

https://leetcode.cn/problems/escape-the-spreading-fire/

Hard # 2023/11/09

```go
func maximumMinutes(grid [][]int) int {
    m := len(grid)
    n := len(grid[0])
    dir := [][]int{{-1, 0}, {0, 1}, {1, 0}, {0, -1}}
    // 火情初始化
    fireTime := make([][]int, m)
    for i := 0; i < m; i++ {
        fireTime[i] = make([]int, n)
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == 1 {
                fireTime[i][j] = 0
            } else {
                fireTime[i][j] = math.MaxInt
            }
        }
    }

    // 计算火烧到每个位置所需时间
    bfs0 := func() {
        visited := make([][]bool, m)
        for i := 0; i < m; i++ {
            visited[i] = make([]bool, n)
        }
        q := [][]int{}
        for i := 0; i < m; i++ {
            for j := 0; j < n; j++ {
                if fireTime[i][j] == 0 {
                    q = append(q, []int{i, j})
                    visited[i][j] = true
                }
            }
        }
        depth := 1
        for len(q) > 0 {
            l := len(q)
            for i := 0; i < l; i++ {
                pos := q[0]
                q = q[1:]
                fireTime[pos[0]][pos[1]] = depth
                for k := 0; k < 4; k++ {
                    x := pos[0] + dir[k][0]
                    y := pos[1] + dir[k][1]
                    if x >= 0 && x < m && y >= 0 && y < n && grid[x][y] != 2 && fireTime[x][y] == math.MaxInt {
                        if !visited[x][y] {
                            visited[x][y] = true
                            q = append(q, []int{x, y})
                        }
                    }
                }
            }
            depth++
        }
    }

    bfs0()

    //  检验人在等待t时间后能否到达终点
    bfs1 := func(t int) bool {
        if fireTime[0][0] <= t {
            return false
        }
        visited := make([][]bool, m)
        for i := 0; i < m; i++ {
            visited[i] = make([]bool, n)
        }
        q := [][]int{{0, 0}}
        visited[0][0] = true
        depth := 1
        for len(q) > 0 {
            l := len(q)
            for i := 0; i < l; i++ {
                pos := q[0]
                q = q[1:]
                if pos[0] == m - 1 && pos[1] == n - 1 {
                    return true
                }
                if fireTime[pos[0]][pos[1]] <= t + depth {
                    continue
                }
                for k := 0; k < 4; k++ {
                    x := pos[0] + dir[k][0]
                    y := pos[1] + dir[k][1]
                    if x >= 0 && x < m && y >= 0 && y < n && grid[x][y] != 2 && fireTime[x][y] >= t + depth + 1 {
                        if !visited[x][y] {
                            visited[x][y] = true
                            q = append(q, []int{x, y})
                        }
                    }
                }
            }
            depth++
        }
        return false
    }
    
    // 二分答案
    rightbound := min(fireTime[0][0] - 1, m * n)
    left, right := 0, rightbound
    for left <= right {
        mid := left + (right - left) / 2
        if bfs1(mid) == true {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    if left == rightbound + 1 {
        return 1e9
    }
    return left - 1
}
```

### 2415. Reverse Odd Levels of Binary Tree

https://leetcode.cn/problems/reverse-odd-levels-of-binary-tree/

Medium # 2023/12/15

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func reverseOddLevels(root *TreeNode) *TreeNode {
    depth := 0
    q := []*TreeNode{root}
    for len(q) > 0 {
        l := len(q)
        if depth % 2 == 1 {
            for i := 0; i < l / 2; i++ { // 换值
                q[i].Val, q[l-1-i].Val = q[l-1-i].Val, q[i].Val
            }
        }
        for i := 0; i < l; i++ {
            node := q[0]
            q = q[1:]
            if node.Left != nil {
                q = append(q, node.Left)
                q = append(q, node.Right)
            }
        }
        depth++
    }
    return root
}
```

### 542. 01 Matrix

https://leetcode.cn/problems/01-matrix/

Medium # 2023/12/19

```go
func updateMatrix(mat [][]int) [][]int {
    r, c := len(mat), len(mat[0])
    dir := [][]int{
        {0, 1},
        {0, -1},
        {1, 0},
        {-1, 0},
    }
    res := make([][]int, r)
    for i := 0; i < r; i++ {
        res[i] = make([]int, c)
        for j := 0; j < c; j++ {
            res[i][j] = math.MaxInt
        }
    }
    visited := make([][]bool, r)
    for i := 0; i < r; i++ {
        visited[i] = make([]bool, c)
    }
    q := []Pair{}
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            if mat[i][j] == 0 {
                res[i][j] = 0
                visited[i][j] = true
                q = append(q, Pair{i, j})
            }
        }
    }
    depth := 1
    for len(q) > 0 {
        l := len(q)
        for i := 0; i < l; i++ {
            node := q[0]
            q = q[1:]
            for k := 0; k < 4; k++ {
                row, col := node.row + dir[k][0], node.col + dir[k][1]
                if row >= 0 && row < r && col >= 0 && col < c && !visited[row][col] {
                    res[row][col] = depth
                    visited[row][col] = true
                    q = append(q, Pair{row, col})
                }
            }
        }
        depth++
    }
    return res
}

type Pair struct {
    row, col int
}
```

### 2641. Cousins in Binary Tree II

https://leetcode.cn/problems/cousins-in-binary-tree-ii/

Medium # 2024/02/07

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func replaceValueInTree(root *TreeNode) *TreeNode {
    root.Val = 0
    q := []*TreeNode{root}
    sums := []int{}
    for len(q) > 0 {
        l := len(q)
        sum := 0
        for i := 0; i < l; i++ {
            node := q[0]
            q = q[1:]
            if node.Left != nil {
                sum += node.Left.Val
                q = append(q, node.Left)
            }
            if node.Right != nil {
                sum += node.Right.Val
                q = append(q, node.Right)
            }
        }
        sums = append(sums, sum)
    }

    q2 := []*TreeNode{root}
    depth := 0
    for len(q2) > 0 {
        l := len(q2)
        for i := 0; i < l; i++ {
            node := q2[0]
            q2 = q2[1:]
            sum := 0
            if node.Left != nil {
                sum += node.Left.Val
                q2 = append(q2, node.Left)
            }
            if node.Right != nil {
                sum += node.Right.Val
                q2 = append(q2, node.Right)
            }
            if node.Left != nil {
                node.Left.Val = sums[depth] - sum
            }
            if node.Right != nil {
                node.Right.Val = sums[depth] - sum
            }
        }
        depth++
    }
    return root
}
```

### 993. Cousins in Binary Tree

https://leetcode.cn/problems/cousins-in-binary-tree/

Simple # 2024/02/08

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func isCousins(root *TreeNode, x int, y int) bool {
    depthMap := make(map[int]int)
    parentMap := make(map[int]int)
    q := []*TreeNode{root}
    depth := 0
    for len(q) > 0 {
        l := len(q)
        for i := 0; i < l; i++ {
            node := q[0]
            q = q[1:]
            depthMap[node.Val] = depth
            if node.Left != nil {
                q = append(q, node.Left)
                parentMap[node.Left.Val] = node.Val
            }
            if node.Right != nil {
                q = append(q, node.Right)
                parentMap[node.Right.Val] = node.Val
            }
        }
        depth++
    }
    if depthMap[x] == depthMap[y] && parentMap[x] != parentMap[y] {
        return true
    } else {
        return false
    }
}
```

