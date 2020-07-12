
# botはツイートを圧縮するとサイズが小さくなることを利用して、botの検出

## 今やっていること
**真面目に相性を考慮した企業推薦アプリやマッチングアプリを作りたい**  
企業への就職や出会いを求める場など、現在はITが進んでいますが、まだ最適な状態に至っていいないだろうと思われます。そんな課題を解決するために、人の行動ログ（ここではSNSでの発信ログ等）を利用して、真面目なマッチングエンジンを作ろうとしていました。  

具体的な多くの人の行動ログを取得可能なサービスを所有していないので、Twitter社のデータを用いて、マッチングエンジンを作ろうとして現在、技術検証や精度の改善などをしています。  

日本語のテキストを書くユーザ **2400万人** 分の直近500 ~ 1000ツイート程度をサンプリングしており、さまざまな観点を検証しています。

**安西先生...、botが邪魔です...!**  
狭い課題として、botと呼ばれるプログラムでの自動運用されたアカウントが少なくない数存在し、botは特定のキーワードを何度もつぶやくので、個性や特性を重要視したマッチングエンジンを作成した際に悪影響を及ぼす可能性が強くあります。  

そのため、狭いスコープの課題ですが、botをデータと機械学習でうまく検出し、ブロックする仕組みを構築しましたので、ご参考にしていただけますと幸いです。  

## 先行研究
奈良先端科学技術大学院大学が数年前に出した軽めの（？）論文に、[認知症者􏰀発言􏰁圧縮すると小さなサイズになる](https://www.jstage.jst.go.jp/article/pjsai/JSAI2016/0/JSAI2016_4D11/_pdf)というものがあります。  

認知症の人は、認知症ではない人に比べて、発話した情報が、圧縮アルゴリズムにて圧縮すると、小さくなるというものでした。  

ハフマン符号化をかけることで、頻出するパターンをより短い符号に置き換えでデータの圧縮サイズをあげようというものになります。  

## pythonで処理するには同系統アルゴリズムであるbzip2が便利  

bzip2は、明確に `ブロックソート法` と `ハフマン符号化` を行っており、かつ、pythonで標準でサポートされている圧縮方式より結果が綺麗に出たので、採用しました。  

具体的には、以下のようなプロセスで元のテキストファイルサイズと、圧縮済みのファイルサイズを比較します。  

```python
import bz2
import os
import random
from pathlib import Path

salt = f"{random.random():0.12f}"
pid = f"{os.getpid()}_{salt}"

all_text # ある特定のユーザのツイートを１つにjoinしたもの

# そのままのテキスト情報を書き込んだときのサイズを取得
with open(f"/tmp/{pid}", "w") as fp:
   fp.write(all_text)
original_size = Path(f"/tmp/{pid}").stat().st_size

# bz2で圧縮した時のサイズを取得
with bz2.open(f"/tmp/{pid}", "wt") as fp:
    fp.write(all_text)
bz2_size = Path(f"/tmp/{pid}").stat().st_size

# clean up
Path(f"/tmp/{pid}").unlink()
```

## 圧縮アルゴリズムで作った特徴量と、特徴量エンジニアリングで作った特徴量を組み合わせて学習

### ラベルの定義
**スクリーンネームの末尾が `_bot` になっているものをbotと定義**  

botはかなりの数存在し、検索を汚染するような形でよく出現します。twitterのスクリーンネームで末尾に_botをつけているアカウントが存在し、これはほぼbotだろうという前提で処理して良いものであろうと理解できます。スクリーンネームの末尾が `_bot` ならば、botと定義しました。 


<div align="center">
    <img width="500px" src="https://user-images.githubusercontent.com/4949982/87244861-76b7cd00-c47b-11ea-9758-e95ccbe76199.png">
</div>
縦軸に `圧縮したファイルサイズ/オリジナルファイルサイズ` , 横軸に `機械的な周期性` を取ると、上記のような散布図を得ることができます。  

**ノイズもままある**  
`_bot` と末尾のスクリーンネームにつけているのにbot出ない人、それなりにいるんです。Twitterをやっていると理解できる事柄ですが、自身のアイデンティティなどを機械などと近い存在であると思っている人は `_bot` とかつけることがあるようです。  
手動でクリーニング仕切れない量程度には、こう言った人なのに、 `_bot` のサフィックスをつけている人がいるのですが、機械学習でうまくやることで解決していきます（又は一部諦める）

<div align="center">
   <img width="500px" src="https://user-images.githubusercontent.com/4949982/87243783-a9f65e00-c473-11ea-91a4-51b9d3248193.png">
</div>

例えば、上記の図の左側でまるで囲まれた `_bot` であると自称しているユーザは、実際は典型的なユーザの使い方をしており、botではありません。 

**`_bot` が末尾のものだけでは、`precision` によりすぎている**  

真の課題では、`_bot` のサフィックス等がつかなくても人間のユーザのフリをするbotアカウントを検出することであります。このアカウントは大量にあり、自動運用されている宣伝・企業アカウントなどはまずbotであると自分では言いません。    

そのため、 `_bot` は例外があるものの、precisionによりすぎた判別をrecall側に倒す作業とも捉えることが可能になります。 

### モデルの閾値を調整し、現実的なリコールになるようにする

#### 特徴量を選定する
今回のスコープとしてrecallを上がるように倒したいので、モデルの複雑度を上げすぎないことと、ノイズとなるbotでないのにbotを自称している人を判別してはいけないので、うまく丸めるため特徴量を多くしすぎないことが課題となります。  

以下の特徴量を選択しました。
 - compression_ratio: 圧縮率
 - is_near_rate: 前のツイートから10分の倍率でツイートした率
 - uniq_ratio: ユニークツイート数


#### モデルの作成
LightGBMで、特徴量を3つ用いて、構築する木の複雑さを3に抑えて、AUCをメトリックとして学習を行いました。 
Holdoutで25%をvalidationとして、AUCが悪化しない範囲で学習を継続します。  
特徴量単体だと、AUCが 0.90程度までしかでませんが、 `0.922` まで上げることが可能になりました。  


####  作成したモデルのしきい値探索
**どの程度、blockしていいものかを定義する**  

`_bot` がつくアカウントがTwitterのアカウントの全体の `0.05%` 程度でした。定性的な主張ですが、2%程度は完全にbotに近いアカウントであり、これらを閾値を0.05まで緩めることで、2%程度の大きなリコールを得ることができました。  

<div align="center">
   <img width="500px" src="https://user-images.githubusercontent.com/4949982/87243823-06597d80-c474-11ea-83a9-f585d242f6fb.png">
</div>

素の `_bot` だけの散布図より、オレンジの面積が大きく広がっていることがわかります。  

#### モデルの結果の訂正評価

（著作権法32条に基づき、技術検証のため、公開情報を引用しています）

**例1: もう使用していないアカウントでアプリ登録をしたアカウントをボットとして判定**  
<div align="center">
   <img width="350px" src="https://user-images.githubusercontent.com/4949982/87244671-c8f7ee80-c479-11ea-8569-a92c585377da.png">
   <div>連携したアプリからの宣伝のみなのでbot判定でOK</div>
</div>

**例2: 出会い系の誘導の業者アカウントをボットとして判定**  
<div align="center">
   <img width="350px" src="https://user-images.githubusercontent.com/4949982/87244672-cc8b7580-c479-11ea-9170-ca4abfae8453.png">
   <div>実態は偽装した業者アカウントbotなので見抜けている</div>
</div>

## Twitter APIでまとめてbotをブロックする
このbotの検出は単体でも価値がある作業で、Twitterの検索結果をbotや業者が激しく汚染する、という経験を体験している方は多いかと存じます。  
検出されたbotを Twitter APIとそれをpythonで簡単に使えるようにした、[tweepy](https://www.tweepy.org/)をインストールすることで、簡単に特定のユーザをblockすることができます。  

書捨てのコードだと以下のようにして実行することができます。  

入力となるデータは末尾のDropboxのリンクに付属します。  
```python
import time
import tweepy
import os
import pandas as pd
from tqdm import tqdm

auth = tweepy.OAuthHandler(os.environ["API_KEY"], os.environ["API_SECRET_KEY"])
auth.set_access_token(os.environ["ACCESS_TOKEN"], os.environ["ACCESS_TOKEN_SECRET"])

api = tweepy.API(auth)

df = pd.read_csv("./tmp/result.csv")
df.sort_values(by=["yhat"], ascending=False, inplace=True)

for username, yhat in tqdm(zip(df.username, df.yhat), desc="blocking...", total=len(df)):
    try:
        api.create_block(username)
        # time.sleep(0.1) # you may need this
    except Exception as exc:
        print(exc)
        continue
```
 

## データとコード
 - [GitHub](https://github.com/GINK03/bot-detect-and-block): 再現をしたい場合、何をやったかが確認できます
 - [Dropbox](https://www.dropbox.com/s/zr10lyj322zhc9o/result_20200712.csv?dl=0): 最終的な、推論したスコアを付与したデータ
 - [Dropbox](https://www.dropbox.com/s/sjlqmw5pk1j1zzu/bot-detect-and-block-snapshot_20200712.zip?dl=0): Githubにあるコードを再現するためのデータ（オリジナルのTweet情報は含みません）

## Webアプリにできないでしょうか？
 - 常に新鮮なコーパスはあり、集計はできる
 - TwitterIDでログインすると、一括でブロックできるアプリは作ることができ、また、世の中に必要な気がします。

