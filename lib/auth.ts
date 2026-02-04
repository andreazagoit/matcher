import NextAuth from "next-auth";

/**
 * Auth.js Configuration
 * 
 * Uses our own OAuth provider (Matcher) for authentication.
 * No separate login - all auth goes through OAuth flow.
 */
export const { handlers, signIn, signOut, auth } = NextAuth({
    providers: [
        {
            id: "matcher",
            name: "Matcher",
            type: "oauth",

            // Our OAuth endpoints
            authorization: {
                url: "http://localhost:3000/oauth/authorize",
                params: {
                    scope: "openid profile email",
                    response_type: "code",
                },
            },
            token: "http://localhost:3000/oauth/token",
            userinfo: "http://localhost:3000/oauth/userinfo",

            // System OAuth app credentials (from seed)
            clientId: process.env.OAUTH_CLIENT_ID!,
            clientSecret: process.env.OAUTH_CLIENT_SECRET!,

            // Map userinfo response to Auth.js user
            profile(profile) {
                return {
                    id: profile.sub,
                    name: profile.name,
                    email: profile.email,
                };
            },
        },
    ],

    session: {
        strategy: "jwt",
        maxAge: 7 * 24 * 60 * 60, // 7 days
    },

    callbacks: {
        async jwt({ token, user, account }) {
            if (user) {
                token.id = user.id;
            }
            // Store access token for API calls
            if (account) {
                token.accessToken = account.access_token;
            }
            return token;
        },
        async session({ session, token }) {
            if (token && session.user) {
                session.user.id = token.id as string;
            }
            return session;
        },
    },

    pages: {
        signIn: "/login",
    },
});
