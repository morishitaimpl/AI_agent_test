# AI_agent_test
APIを使用したAIアプリのテスト

claudeモデルを使用しているので、ANTHROPIC_API_KEYを.envなどの隠しファイルに環境変数を設定してください
別モデルを使用する場合は、select_model等変更する必要があるので注意してください。

生成の質を調整するtemperatureパラメーターを任意で設定できます。

必要なライブラリは次のようにインストールしてください。pip install -r library_install.txt

実行コマンドは次のとおりです。streamlit run agent.py

agent_ReAct.py
ReActパターンを組み込んでみました。
