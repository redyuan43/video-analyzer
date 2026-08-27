import SwiftUI

@main
struct VideoAnalyzerApp: App {

    @StateObject private var browser = BrowserModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(browser)
                .onOpenURL { url in
                    handle(url)
                }
        }
    }

    /// Handles `videoanalyzer://share?url=<encoded>` from the share extension.
    private func handle(_ url: URL) {
        guard url.scheme == "videoanalyzer", url.host == "share" else { return }
        guard let shared = URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?
            .first(where: { $0.name == "url" })?
            .value,
            !shared.isEmpty
        else { return }

        browser.pendingSharedURL = shared
    }
}
