/*
 * @lc app=leetcode.cn id=300 lang=golang
 *
 * [300] 最长递增子序列
 */

// @lc code=start
// 牌堆法
func lengthOfLIS(nums []int) int {
	l := len(nums)
	top := make([]int, l) // 牌堆顶的牌是升序的
	piles := 0
	for i := 0; i < l; i++ {
		// 二分找到牌堆顶中第一个大于当前值的堆
		left, right := 0, piles-1
		for left <= right {
			mid := left + (right-left)/2
			if top[mid] < nums[i] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
		if left == piles {
			piles++
		}
		top[left] = nums[i]
	}
	return piles
}

// @lc code=end

