from preprocess import get_bert_data
from header import timestamp
import torch

LM = "LM/chinese_wwm_pytorch/"

def get_summary(mode, encoder, origin_text):
    summary = ""
    # preprocess to bert_data
    timestamp("preprocess to bert_data")
    bert_data = get_bert_data(origin_text, LM=LM)
    torch.save(bert_data, "dataset/inference/infer.test.pt")
    timestamp("--")

    # feed into model by calling command

    # get output

    print(bert_data)
    return summary

mode = "ext"
encoder = "trans"
origin_text = """歡迎回到新聞現場關心國際焦點 
以 巴 持續 傳出 流血衝突 
不過 以色列 星期二 展開 另 一 波 撤軍 行動 退出 先前 占領 的 巴勒斯坦 政經 中心 拉 瑪 拉 
而 根據 正在 歐洲 的 以色列 外長 表示 他 和 以色列 總理 正在 著手 新 的 中東 和平 計畫 
以色列 星期二 開始 從 巴勒斯坦自治 城市 拉 瑪 拉 撤軍 
不過 在 同 一 天 約旦河西岸 還是 有 零星 衝突 發生 造成 六 個人 死亡 
自從 三 個 星期 前 以色列 觀光 部長 被 人 暗殺 以後 以色列 軍隊 就 陸續 占領 六 個 巴勒斯坦 城鎮 
過去 幾 個 星期 在 美國 的 壓力 之下 以色列 已經 陸續 撤出 伯 力 恆 貝 特 假 拉 和 卡 其 里 雅 三 個 城鎮 
而 在 昨天 從 耶路撒冷 北邊 的 拉 瑪 拉 撤軍 後 以色列 國防部長 表示 他們 還 會 在 拉 瑪 拉 周圍 維持 一個 安全 路障 
拉 瑪 拉 目前 是 巴勒斯坦 當局 的 重要 政治 和 經濟 中心 
在 以色列 軍隊 和 坦克 撤離 的 同時 一些 巴勒斯坦 人 也 在 當地 進行 示威 抗議 
在 以色列 結束 對 拉 瑪 拉長 達 兩 個 星期 的 占領 後 目前 只剩下 節 寧和 土 卡 姆 仍 為 以色列 軍隊 所 占領 
在 星期二 稍早 時 西岸 的 城市 節 寧 有 兩 名 巴勒斯坦 激進份子 所 搭乘 的 車輛 遭到 襲擊 造成 兩 人 死亡 
這 兩 個 當中 的 一 人 一直 受到 以色列 的 通緝 
另外 在 靠近 那 布魯 斯 附近 有 三 名 巴勒斯坦 槍手 和 一 名 以色列 士兵 在 雙方 交戰 時 喪生 
以色列 和 巴勒斯坦 新 一 波 的 緊張 衝突 是 由於 以色列 內閣 部長 在 十月 十七 號 遭到 暗殺 所 引發 
但 由於 美國 擔心 中東地區 的 不 穩定 會 影響到 他 的 反恐 聯盟 因而 向 以 巴 雙方 施壓 要求 停止 暴力 衝突 的 發生 
前往 比利時 布魯塞爾 參加 歐洲 與 地中海 國家 外長 會議 的 以色列 外長 裴瑞 斯 表示 他 已 和 總理夏隆 建構 新 的 中東 和平 計畫 
希望 藉此 終結 雙方 一 年 多 來 造成 超過 九 百 人 喪生 的 流血衝突 
根據 以色列 媒體 的 報導 裴瑞 斯 所 提議 的 新 和平 方案 
包括 一 項 停火 協議 以及 以 聯合國 的 議案 為 基礎 和 巴勒斯坦 人 展開 談判 
在 聯合國 的 決議案 當中 要求 以色列 撤出 一 九 七 六 年 以 阿 戰爭 中 所 占領 的 土地 
同時 建立 一個 巴勒斯坦 國 
公視 新聞 陳 秋 玫 編譯 
"""
get_summary(mode, encoder, origin_text)