"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircleIcon, XCircleIcon, Loader2Icon, CopyIcon } from "lucide-react";

interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  scope?: string;
}

function OAuthCallbackContent() {
  const searchParams = useSearchParams();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  const [tokens, setTokens] = useState<TokenResponse | null>(null);

  const exchangeCodeForTokens = async (code: string, codeVerifier: string, clientId: string) => {
    try {
      const response = await fetch("/api/oauth/token", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: new URLSearchParams({
          grant_type: "authorization_code",
          code,
          redirect_uri: `${window.location.origin}/dashboard/oauth-test-callback`,
          client_id: clientId,
          code_verifier: codeVerifier,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setStatus("error");
        setError(data.error_description || data.error || "Token exchange failed");
        return;
      }

      setTokens(data);
      setStatus("success");

      // Clean up
      sessionStorage.removeItem("oauth_test_state");
      sessionStorage.removeItem("oauth_test_code_verifier");
      sessionStorage.removeItem("oauth_test_client_id");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Token exchange failed");
    }
  };

  const processCallback = () => {
    const code = searchParams.get("code");
    const state = searchParams.get("state");
    const errorParam = searchParams.get("error");
    const errorDescription = searchParams.get("error_description");

    if (errorParam) {
      setStatus("error");
      setError(errorDescription || errorParam);
      return;
    }

    if (!code) {
      setStatus("error");
      setError("No authorization code received");
      return;
    }

    // Verify state
    const storedState = sessionStorage.getItem("oauth_test_state");
    if (state !== storedState) {
      setStatus("error");
      setError("State mismatch - possible CSRF attack");
      return;
    }

    const codeVerifier = sessionStorage.getItem("oauth_test_code_verifier");
    const clientId = sessionStorage.getItem("oauth_test_client_id");

    if (!codeVerifier || !clientId) {
      setStatus("error");
      setError("Missing PKCE code verifier or client ID");
      return;
    }

    // Exchange code for tokens
    exchangeCodeForTokens(code, codeVerifier, clientId);
  };

  useEffect(() => {
    processCallback();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <Card className="w-full max-w-lg">
      <CardHeader className="text-center">
        {status === "loading" && (
          <>
            <Loader2Icon className="h-12 w-12 mx-auto mb-4 animate-spin text-primary" />
            <CardTitle>Exchanging Code for Tokens...</CardTitle>
            <CardDescription>Please wait while we complete the OAuth flow</CardDescription>
          </>
        )}
        {status === "success" && (
          <>
            <CheckCircleIcon className="h-12 w-12 mx-auto mb-4 text-green-500" />
            <CardTitle className="text-green-500">OAuth Test Successful!</CardTitle>
            <CardDescription>The OAuth flow completed successfully</CardDescription>
          </>
        )}
        {status === "error" && (
          <>
            <XCircleIcon className="h-12 w-12 mx-auto mb-4 text-red-500" />
            <CardTitle className="text-red-500">OAuth Test Failed</CardTitle>
            <CardDescription>{error}</CardDescription>
          </>
        )}
      </CardHeader>

      {status === "success" && tokens && (
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Access Token</label>
            <div className="relative">
              <pre className="bg-muted p-3 rounded-lg text-xs font-mono overflow-x-auto pr-10">
                {tokens.access_token}
              </pre>
              <Button
                variant="ghost"
                size="sm"
                className="absolute top-1 right-1 h-8 w-8 p-0"
                onClick={() => copyToClipboard(tokens.access_token)}
              >
                <CopyIcon className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {tokens.refresh_token && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Refresh Token</label>
              <div className="relative">
                <pre className="bg-muted p-3 rounded-lg text-xs font-mono overflow-x-auto pr-10">
                  {tokens.refresh_token}
                </pre>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute top-1 right-1 h-8 w-8 p-0"
                  onClick={() => copyToClipboard(tokens.refresh_token!)}
                >
                  <CopyIcon className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Token Type:</span>
              <span className="ml-2 text-foreground">{tokens.token_type}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Expires In:</span>
              <span className="ml-2 text-foreground">{tokens.expires_in}s</span>
            </div>
            {tokens.scope && (
              <div className="col-span-2">
                <span className="text-muted-foreground">Scopes:</span>
                <span className="ml-2 text-foreground">{tokens.scope}</span>
              </div>
            )}
          </div>

          <Button className="w-full mt-4" onClick={() => window.close()}>
            Close Window
          </Button>
        </CardContent>
      )}

      {status === "error" && (
        <CardContent>
          <Button variant="outline" className="w-full" onClick={() => window.close()}>
            Close Window
          </Button>
        </CardContent>
      )}
    </Card>
  );
}

function LoadingFallback() {
  return (
    <Card className="w-full max-w-lg">
      <CardHeader className="text-center">
        <Loader2Icon className="h-12 w-12 mx-auto mb-4 animate-spin text-primary" />
        <CardTitle>Loading...</CardTitle>
        <CardDescription>Initializing OAuth callback</CardDescription>
      </CardHeader>
    </Card>
  );
}

export default function OAuthTestCallbackPage() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <Suspense fallback={<LoadingFallback />}>
        <OAuthCallbackContent />
      </Suspense>
    </div>
  );
}

