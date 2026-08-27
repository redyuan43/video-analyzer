import UIKit
import WebKit

/// Hosts the analyzer web UI.
///
/// This is a UIKit controller rather than a plain `UIViewRepresentable` because
/// the pieces that make a web app feel native — a refresh control on the
/// scroll view, presenting a share sheet for downloads — all want a real view
/// controller to hang off.
final class AnalyzerWebViewController: UIViewController {

    /// Reported back to SwiftUI so the chrome can show progress and failures.
    var onLoadingChanged: ((Bool) -> Void)?
    var onFailure: ((String?) -> Void)?

    private(set) var webView: WKWebView!
    private let refreshControl = UIRefreshControl()
    private var progressObservation: NSKeyValueObservation?

    /// Base URL of the analyzer. Changing it reloads from the new host.
    var serverURL: URL {
        didSet {
            guard serverURL != oldValue else { return }
            loadHome()
        }
    }

    init(serverURL: URL) {
        self.serverURL = serverURL
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) is not used")
    }

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        setUpWebView()
        loadHome()
    }

    private func setUpWebView() {
        let configuration = WKWebViewConfiguration()

        // The console embeds video players and plays generated narration audio.
        // Without these, iOS forces fullscreen playback and blocks autoplay,
        // which breaks the timestamp-seeking flow in the Q&A view.
        configuration.allowsInlineMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []

        // Persistent store, so a reopened app keeps whatever the UI put in
        // localStorage instead of resetting to a cold console every launch.
        configuration.websiteDataStore = .default()

        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsBackForwardNavigationGestures = true

        // The web layer positions itself against the safe area via
        // `viewport-fit=cover` and `env(safe-area-inset-*)`, so let it own the
        // full screen rather than insetting it twice.
        webView.scrollView.contentInsetAdjustmentBehavior = .never

        refreshControl.addTarget(self, action: #selector(handleRefresh), for: .valueChanged)
        webView.scrollView.refreshControl = refreshControl

        // WKWebView reports load progress but not "finished" for SPA-style
        // in-page updates; the navigation delegate covers the rest.
        progressObservation = webView.observe(\.estimatedProgress, options: [.new]) { [weak self] webView, _ in
            if webView.estimatedProgress >= 1.0 {
                self?.refreshControl.endRefreshing()
            }
        }

        webView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(webView)
        NSLayoutConstraint.activate([
            webView.topAnchor.constraint(equalTo: view.topAnchor),
            webView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            webView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        ])
    }

    // MARK: - Navigation

    func loadHome() {
        load(url: serverURL)
    }

    /// Opens the console with a URL prefilled, as handed over by the share
    /// extension. `mobile.js` picks up the `share` parameter.
    func loadSharedVideoURL(_ videoURL: String) {
        guard var components = URLComponents(url: serverURL, resolvingAgainstBaseURL: false) else {
            return
        }
        components.queryItems = [URLQueryItem(name: "share", value: videoURL)]
        guard let url = components.url else { return }
        load(url: url)
    }

    private func load(url: URL) {
        onFailure?(nil)
        // The analyzer is a live dashboard; a stale cached shell would show
        // job state that no longer exists.
        webView.load(URLRequest(url: url, cachePolicy: .reloadRevalidatingCacheData))
    }

    func reload() {
        if webView.url == nil {
            loadHome()
        } else {
            webView.reload()
        }
    }

    @objc private func handleRefresh() {
        reload()
    }

    /// True when there is somewhere to go back to inside the web UI.
    var canGoBack: Bool { webView.canGoBack }

    func goBack() {
        webView.goBack()
    }
}

// MARK: - WKNavigationDelegate

extension AnalyzerWebViewController: WKNavigationDelegate {

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        onLoadingChanged?(true)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        onLoadingChanged?(false)
        refreshControl.endRefreshing()
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        finishWithError(error)
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
    ) {
        finishWithError(error)
    }

    private func finishWithError(_ error: Error) {
        onLoadingChanged?(false)
        refreshControl.endRefreshing()

        // Cancelling one navigation by starting another is normal, not a failure.
        let nsError = error as NSError
        guard !(nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled) else {
            return
        }
        onFailure?(error.localizedDescription)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        // Report artifacts, PDFs and media the web view cannot render are
        // downloads, not dead ends.
        if navigationResponse.canShowMIMEType {
            decisionHandler(.allow)
        } else {
            decisionHandler(.download)
        }
    }

    func webView(
        _ webView: WKWebView,
        navigationResponse: WKNavigationResponse,
        didBecome download: WKDownload
    ) {
        download.delegate = self
    }

    func webView(
        _ webView: WKWebView,
        navigationAction: WKNavigationAction,
        didBecome download: WKDownload
    ) {
        download.delegate = self
    }
}

// MARK: - WKUIDelegate

extension AnalyzerWebViewController: WKUIDelegate {

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        guard let url = navigationAction.request.url else { return nil }

        // `target="_blank"` is used for two different intents in the UI:
        // opening one of our own artifacts, and opening the original video on
        // its source site. Keep ours in-app and hand the rest to Safari.
        if url.host == serverURL.host {
            webView.load(navigationAction.request)
        } else {
            UIApplication.shared.open(url)
        }
        return nil
    }

    // WKWebView drops JS dialogs on the floor unless the host app renders them.
    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping () -> Void
    ) {
        let alert = UIAlertController(title: nil, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "好", style: .default) { _ in completionHandler() })
        present(alert, animated: true)
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping (Bool) -> Void
    ) {
        let alert = UIAlertController(title: nil, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "取消", style: .cancel) { _ in completionHandler(false) })
        alert.addAction(UIAlertAction(title: "确定", style: .default) { _ in completionHandler(true) })
        present(alert, animated: true)
    }
}

// MARK: - WKDownloadDelegate

extension AnalyzerWebViewController: WKDownloadDelegate {

    func download(
        _ download: WKDownload,
        decideDestinationUsing response: URLResponse,
        suggestedFilename: String,
        completionHandler: @escaping (URL?) -> Void
    ) {
        // A unique subdirectory avoids clobbering a previous download that
        // shares the server's suggested filename.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("downloads/\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            completionHandler(nil)
            return
        }
        completionHandler(directory.appendingPathComponent(suggestedFilename))
    }

    func downloadDidFinish(_ download: WKDownload) {
        guard let url = download.progress.fileURL else { return }
        // Hand off to the system share sheet so the file can land in Files,
        // Notes, or wherever the user actually wants it.
        let picker = UIActivityViewController(activityItems: [url], applicationActivities: nil)
        picker.popoverPresentationController?.sourceView = view
        picker.popoverPresentationController?.sourceRect = CGRect(
            x: view.bounds.midX, y: view.bounds.maxY, width: 0, height: 0
        )
        present(picker, animated: true)
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        onFailure?("下载失败：\(error.localizedDescription)")
    }
}
