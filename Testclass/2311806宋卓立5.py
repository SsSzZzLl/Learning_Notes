def calculate_bonus(profit):
    if profit <= 100000:
        bonus = profit * 0.1
    elif profit <= 200000:
        bonus = 100000 * 0.1 + (profit - 100000) * 0.075
    elif profit <= 400000:
        bonus = 100000 * 0.1 + 100000 * 0.075 + (profit - 200000) * 0.05
    elif profit <= 600000:
        bonus = 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + (profit - 400000) * 0.03
    elif profit <= 1000000:
        bonus = 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + 200000 * 0.03 + (profit - 600000) * 0.015
    else:
        bonus = 100000 * 0.1 + 100000 * 0.075 + 200000 * 0.05 + 200000 * 0.03 + 400000 * 0.015 + (profit - 1000000) * 0.01
    return bonus

# 测试
profit = float(input("请输入利润（单位：万元）："))
total_bonus = calculate_bonus(profit)
print("应发奖金总数为：{} 万元".format(total_bonus))
