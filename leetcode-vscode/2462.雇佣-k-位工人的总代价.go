/*
 * @lc app=leetcode.cn id=2462 lang=golang
 *
 * [2462] 雇佣 K 位工人的总代价
 */

// @lc code=start
func totalCost(costs []int, k int, candidates int) int64 {
	l := len(costs)
	left, right := candidates-1, l-candidates
	h := &Heap{}
	heap.Init(h)
	if left+1 < right {
		for i := 0; i <= left; i++ {
			heap.Push(h, []int{i, costs[i]})
		}
		for i := right; i < l; i++ {
			heap.Push(h, []int{i, costs[i]})
		}
	} else {
		for i := 0; i < l; i++ {
			heap.Push(h, []int{i, costs[i]})
		}
	}

	res := int64(0)
	for i := 0; i < k; i++ {
		min := heap.Pop(h).([]int)
		idx, cost := min[0], min[1]
		res += int64(cost)
		if left+1 < right {
			if idx <= left {
				left++
				heap.Push(h, []int{left, costs[left]})
			} else {
				right--
				heap.Push(h, []int{right, costs[right]})
			}
		}
	}
	return res
}

type Heap [][]int

func (h Heap) Len() int {
	return len(h)
}

func (h Heap) Less(i, j int) bool {
	if h[i][1] == h[j][1] {
		return h[i][0] < h[j][0]
	}
	return h[i][1] < h[j][1]
}

func (h Heap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *Heap) Push(x interface{}) {
	*h = append(*h, x.([]int))
}

func (h *Heap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

// @lc code=end

