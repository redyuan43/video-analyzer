import Foundation

/// Minimal client for the analyzer REST API.
///
/// The app itself talks to the server through the embedded web UI, so this
/// only covers what native code needs: a health probe for the settings screen
/// and job creation for the share extension.
struct VideoAnalyzerClient {

    let baseURL: URL

    init(baseURL: URL = AppConfig.serverURL) {
        self.baseURL = baseURL
    }

    enum ClientError: LocalizedError {
        case badResponse
        case server(status: Int, message: String)

        var errorDescription: String? {
            switch self {
            case .badResponse:
                return "服务器返回了无法解析的响应"
            case let .server(status, message):
                return message.isEmpty ? "服务器返回 HTTP \(status)" : message
            }
        }
    }

    struct CreatedJob {
        let id: String
        let title: String?
        let count: Int
    }

    /// Checks that the configured address is actually an analyzer instance,
    /// not just something answering on that port.
    func checkHealth() async throws -> Bool {
        let request = URLRequest(
            url: baseURL.appending(path: "/api/video-link/health"),
            timeoutInterval: 10
        )
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.badResponse }
        guard http.statusCode == 200 else {
            throw ClientError.server(status: http.statusCode, message: Self.message(from: data))
        }
        let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return payload?["ok"] as? Bool ?? false
    }

    /// Creates a job from a video URL.
    ///
    /// `autoStart` is passed through explicitly rather than defaulted on,
    /// because starting a run kicks off a long GPU pipeline.
    func createJob(videoURL: String, autoStart: Bool) async throws -> CreatedJob {
        let expandBilibiliParts = Self.isBilibiliVideoURL(videoURL)
        let path = expandBilibiliParts
            ? "/api/video-link/jobs/batch"
            : "/api/video-link/jobs"
        let body: [String: Any] = expandBilibiliParts
            ? [
                "video_urls": [videoURL],
                "expand_bilibili_parts": true,
                "auto_start": autoStart,
            ]
            : [
                "video_url": videoURL,
                "auto_start": autoStart,
            ]
        var request = URLRequest(
            url: baseURL.appending(path: path),
            timeoutInterval: expandBilibiliParts ? 150 : 30
        )
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw ClientError.badResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw ClientError.server(status: http.statusCode, message: Self.message(from: data))
        }

        guard let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ClientError.badResponse
        }
        if let id = payload["job_id"] as? String {
            return CreatedJob(id: id, title: payload["title"] as? String, count: 1)
        }
        guard let jobs = payload["jobs"] as? [[String: Any]],
              let first = jobs.first,
              let id = first["job_id"] as? String else {
            throw ClientError.badResponse
        }
        let count = (payload["created"] as? NSNumber)?.intValue ?? jobs.count
        let collection = payload["collection"] as? [String: Any]
        return CreatedJob(
            id: id,
            title: collection?["title"] as? String ?? first["title"] as? String,
            count: count
        )
    }

    /// The API reports failures as `{"error": "..."}`; surface that instead of
    /// a bare status code.
    private static func message(from data: Data) -> String {
        guard let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = payload["error"] as? String else {
            return ""
        }
        return error
    }

    private static func isBilibiliVideoURL(_ value: String) -> Bool {
        guard let url = URL(string: value),
              let host = url.host?.lowercased() else {
            return false
        }
        if host == "b23.tv" {
            return true
        }
        let isBilibili = host == "bilibili.com" || host.hasSuffix(".bilibili.com")
        return isBilibili && url.path.lowercased().hasPrefix("/video/")
    }
}
