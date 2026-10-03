def tests():
    def check(li, expected):
        assert len(Split(li)) == len(expected)
        if li == []:
            return
        for ind in range(len(Split(li))):
            assert Split(li)[ind] == expected[ind]
    check([1,2,3,4,5], [1, 3, 5])
    check(['a', 'b'], ['a', 'b'])
    check([3, -3, 1, -1, 5, -5, 7, -7, 9, -9], [3, -3, 1, -1, 5, -5, 7, -7, 9, -9])
check()
