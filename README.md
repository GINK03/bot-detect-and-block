
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
 - スクリーンネームの末尾 `_bot` がなっているものを定義
 - ノイズもままある
 - `_bot` が末尾のものだけでは、`precision` によりすぎている

### ラベルを拡張して、現実的になリコールに拡張する
 - 作成したモデルのしきい値探索
 - どの程度、blockしていいものかを定義する

## Twitter APIでまとめてbotをブロックする
 - Twitter APIを用いることで、ブロックできる
 - Tweepyを利用すると早い

## データとコード
 - Dropbox
 - GitHub

## Webアプリにできないでしょうか？
 - 新鮮なコーパスはあり、集計はできる
 - Webアプリにしたいんですが、どなたか助けて

