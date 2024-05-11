/*
 * @lc app=leetcode.cn id=1463 lang=golang
 *
 * [1463] 摘樱桃 II
 */

// @lc code=start
func cherryPickup(grid [][]int) int {
	r := len(grid)
	c := len(grid[0])
	dp := make([][][]int, r) // x, y1, y2
	for i := 0; i < r; i++ {
		dp[i] = make([][]int, c)
		for j := 0; j < c; j++ {
			dp[i][j] = make([]int, c)
		}
	}
	dp[0][0][c-1] = grid[0][0] + grid[0][c-1]
	for y1 := 0; y1 < c; y1++ {
		for y2 := 0; y2 < c; y2++ {
			if y1 != 0 || y2 != c-1 {
				dp[0][y1][y2] = -1
			}
		}
	}
	dy := []int{-1, 0, 1}
	for x := 1; x < r; x++ {
		for y1 := 0; y1 < c; y1++ {
			for y2 := 0; y2 < y1; y2++ {
				dp[x][y1][y2] = -1
			}
			for y2 := y1; y2 < c; y2++ {
				way := false
				for i := 0; i < 3; i++ {
					oldy1 := y1 + dy[i]
					if oldy1 < 0 || oldy1 >= c {
						continue
					}
					for j := 0; j < 3; j++ {
						oldy2 := y2 + dy[j]
						if oldy2 < 0 || oldy2 >= c {
							continue
						}
						if dp[x-1][oldy1][oldy2] != -1 {
							dp[x][y1][y2] = max(dp[x][y1][y2], dp[x-1][oldy1][oldy2])
							way = true
						}
					}
				}
				if !way {
					dp[x][y1][y2] = -1
				} else {
					if y1 == y2 {
						dp[x][y1][y2] += grid[x][y1]
					} else {
						dp[x][y1][y2] += grid[x][y1] + grid[x][y2]
					}
				}
			}
		}
	}
	res := 0
	for y1 := 0; y1 < c; y1++ {
		for y2 := y1; y2 < c; y2++ {
			res = max(res, dp[r-1][y1][y2])
		}
	}
	return res
}

// @lc code=end

