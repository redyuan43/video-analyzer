"""video-link 状态引擎包：job 生命周期、阶段编排与视频链接分析调度。

原为 tools/video_link_status_server.py（入口/运维层），Phase 3 迁入正式包，
使 UI 与未来入口可通过 video_analyzer.jobengine 直接导入，消除对 tools/ 的反向依赖。
"""
