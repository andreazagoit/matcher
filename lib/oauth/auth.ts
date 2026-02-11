import NextAuth from "next-auth";

const baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3000";

/**
 * Auth.js Configuration
 * 
 * Configures the authentication system using a custom OAuth provider.
 * The authorization endpoint is handled by our internal /oauth/authorize route.
 */
export const { handlers, signIn, signOut, auth } = NextAuth({
    providers: [
        {
            id: "matcher",
            name: "Matcher",
            type: "oauth",

            // OAuth endpoints (absolute URLs required by Auth.js)
            authorization: {
                url: `${baseUrl}/oauth/authorize`,
                params: {
                    scope: "openid profile email",
                    response_type: "code",
                },
            },
            token: `${baseUrl}/oauth/api/token`,
            userinfo: `${baseUrl}/oauth/api/userinfo`,

            // System OAuth application credentials
            clientId: process.env.OAUTH_CLIENT_ID!,
            clientSecret: process.env.OAUTH_CLIENT_SECRET!,

            /**
             * Maps the userinfo response from our provider to the Auth.js user object.
             */
            profile(profile) {
                return {
                    id: profile.sub,
                    name: profile.name,
                    email: profile.email,
                    firstName: profile.given_name,
                    lastName: profile.family_name,
                    birthDate: profile.birthdate,
                    gender: profile.gender,
                    createdAt: profile.created_at,
                    updatedAt: profile.updated_at,
                } as any;
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
                token.firstName = (user as any).firstName;
                token.lastName = (user as any).lastName;
                token.email = user.email;
                token.birthDate = (user as any).birthDate;
                token.gender = (user as any).gender;
                token.createdAt = (user as any).createdAt;
                token.updatedAt = (user as any).updatedAt;
            }
            if (account) {
                token.accessToken = account.access_token;
            }
            return token;
        },
        async session({ session, token }) {
            if (token && session.user) {
                session.user.id = token.id as string;
                (session.user as any).firstName = token.firstName;
                (session.user as any).lastName = token.lastName;
                (session.user as any).birthDate = token.birthDate;
                (session.user as any).gender = token.gender;
                (session.user as any).createdAt = token.createdAt;
                (session.user as any).updatedAt = token.updatedAt;
            }
            return session;
        },
    },

    trustHost: true,
});
