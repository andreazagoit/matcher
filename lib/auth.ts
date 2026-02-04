import NextAuth from "next-auth";

const baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3000";

/**
 * Auth.js Configuration
 * 
 * Uses our own OAuth provider (Matcher) for authentication.
 * The authorization endpoint (/oauth/authorize) handles login directly.
 */
export const { handlers, signIn, signOut, auth } = NextAuth({
    providers: [
        {
            id: "matcher",
            name: "Matcher",
            type: "oauth",

            // Our OAuth endpoints (relative to baseUrl)
            authorization: {
                url: `${baseUrl}/oauth/authorize`,
                params: {
                    scope: "openid profile email",
                    response_type: "code",
                },
            },
            token: `${baseUrl}/oauth/token`,
            userinfo: `${baseUrl}/oauth/userinfo`,

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

    // Don't redirect to a separate login page - authorize page handles login
    // pages: {
    //     signIn: "/login",
    // },
});
