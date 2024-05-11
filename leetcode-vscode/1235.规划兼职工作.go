/*
 * @lc app=leetcode.cn id=1235 lang=golang
 *
 * [1235] 规划兼职工作
 */

// @lc code=start
func jobScheduling(startTime []int, endTime []int, profit []int) int {
	l := len(startTime)
	jobs := make([][]int, l)
	for i := 0; i < l; i++ {
		jobs[i] = []int{startTime[i], endTime[i], profit[i]}
	}
	sort.Slice(jobs, func(i, j int) bool { // 按结束时间升序排序
		return jobs[i][1] < jobs[j][1]
	})
	dp := make([]int, l+1) // 从前i个工作里选能获得的最大收益
	dp[0] = 0
	for i := 1; i <= l; i++ {
		k := upperBound(jobs, jobs[i-1][0]) // endTime[k] <= startTime[i-1]
		dp[i] = max(dp[i-1], dp[k+1]+jobs[i-1][2])
	}
	return dp[l]
}

func upperBound(arr [][]int, target int) int {
	l, r := 0, len(arr)-1
	for l <= r {
		mid := l + (r-l)/2
		if arr[mid][1] <= target {
			l = mid + 1
		} else {
			r = mid - 1
		}
	}
	return r
}

// @lc code=end

