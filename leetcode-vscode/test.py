import bisect

nums = [0, 2, 2, 4, 4, 4, 6, 6, 8]

print(bisect.bisect_left(nums, -1))
print(bisect.bisect_left(nums, 3))
print(bisect.bisect_left(nums, 4))
print(bisect.bisect_left(nums, 5))
print(bisect.bisect_left(nums, 9))

print(bisect.bisect_right(nums, -1))
print(bisect.bisect_right(nums, 3))
print(bisect.bisect_right(nums, 4))
print(bisect.bisect_right(nums, 5))
print(bisect.bisect_right(nums, 9))