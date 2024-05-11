/*
 * @lc app=leetcode.cn id=2105 lang=golang
 *
 * [2105] 给植物浇水 II
 */

// @lc code=start
func minimumRefill(plants []int, capacityA int, capacityB int) int {
	l := len(plants)
	left, right := 0, l-1
	a, b := capacityA, capacityB
	res := 0
	for left < right {
		if a < plants[left] {
			a = capacityA
			res++
		}
		a -= plants[left]
		left++
		if b < plants[right] {
			b = capacityB
			res++
		}
		b -= plants[right]
		right--
	}
	if left == right {
		if max(a, b) < plants[left] {
			res++
		}
	}
	return res
}

// @lc code=end

