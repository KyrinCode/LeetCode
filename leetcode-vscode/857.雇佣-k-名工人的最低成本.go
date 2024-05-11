/*
 * @lc app=leetcode.cn id=857 lang=golang
 *
 * [857] 雇佣 K 名工人的最低成本
 */

// @lc code=start
func mincostToHireWorkers(quality []int, wage []int, k int) float64 {
	l := len(quality)
	workers := make([]Worker, l)
	for i := 0; i < l; i++ {
		workers[i] = Worker{
			i, quality[i], wage[i],
			float64(wage[i]) / float64(quality[i]),
		}
	}
	sort.Slice(workers, func(i, j int) bool {
		return workers[i].price < workers[j].price
	})

	sum, res := 0, float64(0)

	h := Heap{}
	heap.Init(&h)
	for i := 0; i < l; i++ {
		if h.Len() < k { // res仅有总quality和price决定
			sum += workers[i].quality
			heap.Push(&h, workers[i])
			res = workers[i].price * float64(sum) // price递增
		} else {
			if workers[i].quality < h[0].quality { // price是递增的，只有quality更小才有可能
				sum = sum - h[0].quality + workers[i].quality
				heap.Pop(&h)
				heap.Push(&h, workers[i])
				res = min(res, workers[i].price*float64(sum))
			}
		}
	}
	return res
}

type Worker struct {
	idx, quality, wage int
	price              float64
}

type Heap []Worker

func (h Heap) Len() int {
	return len(h)
}

func (h Heap) Less(i, j int) bool { // 大顶堆
	return h[i].quality > h[j].quality
}

func (h Heap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *Heap) Push(x interface{}) {
	*h = append(*h, x.(Worker))
}

func (h *Heap) Pop() interface{} {
	x := (*h)[len(*h)-1]
	*h = (*h)[:len(*h)-1]
	return x
}

// @lc code=end

