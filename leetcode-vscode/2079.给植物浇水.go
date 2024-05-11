/*
 * @lc app=leetcode.cn id=2079 lang=golang
 *
 * [2079] 给植物浇水
 */

// @lc code=start
func wateringPlants(plants []int, capacity int) int {
	l := len(plants)
	sum := 0
	res := 0
	for i := 0; i < l; i++ {
		if sum+plants[i] <= capacity {
			sum += plants[i]
		} else {
			sum = plants[i]
			res += 2 * i
		}
	}
	res += l
	return res
}

// @lc code=end

