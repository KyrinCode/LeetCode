/*
 * @lc app=leetcode.cn id=354 lang=golang
 *
 * [354] 俄罗斯套娃信封问题
 */

// @lc code=start
func maxEnvelopes(envelopes [][]int) int {
	sort.Slice(envelopes, func(i, j int) bool {
		if envelopes[i][0] == envelopes[j][0] {
			return envelopes[i][1] > envelopes[j][1]
		}
		return envelopes[i][0] < envelopes[j][0]
	}) // 宽升序 同宽则按照高降序
	// 此后只需要对高找到最大递增子序列即可
	l := len(envelopes)
	// dp := make([]int, l) // 以i为结尾的最大递增子序列
	// for i := 0; i < l; i++ {
	// 	dp[i] = 1
	// 	for j := 0; j < i; j++ {
	// 		if envelopes[j][1] > envelopes[i][1] {
	// 			dp[i] = max(dp[i], dp[j]+1)
	// 		}
	// 	}
	// }
	// res := 0
	// for i := 0; i < l; i++ {
	// 	res = max(res, dp[i])
	// }
	// return res
	top := make([]int, l)
	piles := 0
	for i := 0; i < l; i++ {
		// left, right := 0, piles-1
		// for left <= right {
		// 	mid := left + (right-left)/2
		// 	if top[mid] >= envelopes[i][1] { // 第一个大于等于当前值的堆顶
		// 		right = mid - 1
		// 	} else {
		// 		left = mid + 1
		// 	}
		// }
		left := sort.SearchInts(top[:piles], envelopes[i][1])
		if left == piles {
			piles++
		}
		top[left] = envelopes[i][1]
	}
	return piles
}

// @lc code=end

