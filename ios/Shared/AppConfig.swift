import Foundation

/// Server configuration shared between the app and the share extension.
///
/// Both targets read the App Group identifier and the compiled-in default
/// server URL from their Info.plist, which Xcode fills from Config/App.xcconfig.
/// That keeps the bundle prefix in exactly one place instead of being spelled
/// out again in Swift, entitlements and the extension.
enum AppConfig {

    /// Reads a build-setting-backed string out of the running bundle's Info.plist.
    private static func infoValue(_ key: String) -> String? {
        guard let raw = Bundle.main.object(forInfoDictionaryKey: key) as? String else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static var appGroupIdentifier: String {
        infoValue("AppGroupIdentifier") ?? ""
    }

    /// Falls back to `.standard` so a misconfigured App Group degrades into
    /// "the app still works, the extension just cannot see its settings"
    /// rather than a crash.
    static var defaults: UserDefaults {
        let group = appGroupIdentifier
        if !group.isEmpty, let shared = UserDefaults(suiteName: group) {
            return shared
        }
        return .standard
    }

    private static let serverURLKey = "serverURL"

    /// Scheme and host are stored separately in Info.plist and joined here.
    ///
    /// A full URL cannot live in the xcconfig that feeds them: `//` starts a
    /// comment there, so `https://host` would arrive as `https:host` and fail
    /// to parse, leaving the settings screen with every button disabled.
    static var defaultServerURL: URL {
        let scheme = infoValue("DefaultServerScheme") ?? "https"
        if let host = infoValue("DefaultServerHost"),
           let url = URL(string: "\(scheme)://\(host)") {
            return url
        }
        // Only reachable if App.xcconfig was emptied out; keep it valid so the
        // settings screen has something to show and edit.
        return URL(string: "http://127.0.0.1:5000")!
    }

    /// The analyzer base URL, as configured by the user or compiled in.
    static var serverURL: URL {
        get {
            guard let raw = defaults.string(forKey: serverURLKey),
                  let url = URL(string: raw) else {
                return defaultServerURL
            }
            return url
        }
        set {
            defaults.set(newValue.absoluteString, forKey: serverURLKey)
        }
    }

    /// Normalises whatever the user typed into a usable base URL.
    ///
    /// Trailing slashes are dropped so callers can append paths without
    /// doubling the separator. Scheme-less addresses use HTTP unless they are
    /// an HTTPS Tailscale Serve name or use a known HTTPS port.
    static func normalizeServerURL(_ text: String) -> URL? {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if !trimmed.contains("://") {
            trimmed = "\(inferredScheme(for: trimmed))://\(trimmed)"
        }
        while trimmed.hasSuffix("/") {
            trimmed.removeLast()
        }

        guard let url = URL(string: trimmed), let host = url.host, !host.isEmpty else {
            return nil
        }
        return url
    }

    private static func inferredScheme(for text: String) -> String {
        guard let components = URLComponents(string: "http://\(text)"),
              let host = components.host?.lowercased() else {
            return "http"
        }
        if let port = components.port {
            return port == 443 || port == 5443 ? "https" : "http"
        }
        if host.hasSuffix(".ts.net") {
            return "https"
        }
        return "http"
    }
}
