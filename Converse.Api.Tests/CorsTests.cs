using FluentAssertions;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Converse.Api.Tests;

public class CorsTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;
    public CorsTests(WebApplicationFactory<Program> factory) => _factory = factory;

    [Fact]
    public async Task Cross_origin_request_gets_allow_origin_header()
    {
        var client = _factory.CreateClient();
        var req = new HttpRequestMessage(HttpMethod.Get, "/health");
        req.Headers.Add("Origin", "chrome-extension://abcdefghijklmnop");

        var resp = await client.SendAsync(req);

        resp.Headers.Contains("Access-Control-Allow-Origin").Should().BeTrue();
    }
}
