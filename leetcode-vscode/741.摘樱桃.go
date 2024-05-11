/*
 * @lc app=leetcode.cn id=741 lang=golang
 *
 * [741] 摘樱桃
 */

// @lc code=start
func cherryPickup(grid [][]int) int {
	// 转化为两个人一起从(0,0)触发到(n-1,n-1)的最大樱桃值
	n := len(grid)
	dp := make([][][]int, 2*n-1) // k,x1,x2 k=x+y
	for k := 0; k < 2*n-1; k++ {
		dp[k] = make([][]int, n)
		for x1 := 0; x1 < n; x1++ {
			dp[k][x1] = make([]int, n)
			for x2 := 0; x2 < n; x2++ {
				dp[k][x1][x2] = math.MinInt
			}
		}
	}
	dp[0][0][0] = grid[0][0]
	for k := 1; k < 2*n-1; k++ {
		for x1 := max(0, k-n+1); x1 < min(n, k+1); x1++ {
			for x2 := x1; x2 < min(n, k+1); x2++ {
				if grid[x1][k-x1] == -1 || grid[x2][k-x2] == -1 {
					dp[k][x1][x2] = math.MinInt
				} else {
					if x1 == x2 {
						dp[k][x1][x2] = grid[x1][k-x1]
					} else {
						dp[k][x1][x2] = grid[x1][k-x1] + grid[x2][k-x2]
					}
					past := dp[k-1][x1][x2]
					if x1-1 >= 0 && x2-1 >= 0 {
						past = max(past, dp[k-1][x1-1][x2-1])
					}
					if x1-1 >= 0 {
						past = max(past, dp[k-1][x1-1][x2])
					}
					if x2-1 >= 0 {
						past = max(past, dp[k-1][x1][x2-1])
					}
					dp[k][x1][x2] += past
				}
			}
		}
	}
	if dp[2*n-2][n-1][n-1] <= 0 {
		return 0
	}
	return dp[2*n-2][n-1][n-1]
}

// @lc code=end

