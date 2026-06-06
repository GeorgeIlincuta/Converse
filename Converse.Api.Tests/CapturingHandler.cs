namespace Converse.Api.Tests;

internal sealed class CapturingHandler : HttpMessageHandler
{
    public HttpRequestMessage? CapturedRequest { get; private set; }
    public string? CapturedRequestBody { get; private set; }
    public Uri? CapturedRequestUri { get; private set; }
    public System.Net.Http.Headers.AuthenticationHeaderValue? CapturedAuthorizationHeader { get; private set; }
    public HttpMethod? CapturedHttpMethod { get; private set; }
    public required HttpResponseMessage Response { get; set; }

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken cancellationToken)
    {
        CapturedRequest = request;
        CapturedRequestUri = request.RequestUri;
        CapturedAuthorizationHeader = request.Headers.Authorization;
        CapturedHttpMethod = request.Method;
        if (request.Content is not null)
            CapturedRequestBody = await request.Content.ReadAsStringAsync(cancellationToken);
        return Response;
    }
}
