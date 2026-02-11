import NextAuth from "next-auth";

const baseUrl = process.env.NEXTAUTH_URL || "http://localhost:3000";


/**
 * Maps the userinfo response from our provider to the Auth.js user object.
 */
interface MatcherProfile {
    sub: string;
    name: string;
    email: string;
    given_name: string;
    family_name: string;
    birthdate: string;
    gender: string | null;
    created_at: string;
    updated_at: string;
    picture?: string | null;
}

interface MatcherUser {
    id: string;
    name: string;
    email: string;
    firstName: string;
    lastName: string;
    birthDate: string;
    gender: string | null;
    createdAt: string;
    updatedAt: string;
    image?: string | null;
}

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

            profile(profile: MatcherProfile) {
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
                    image: profile.picture,
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
                const u = user as unknown as MatcherUser;
                token.id = user.id;
                token.firstName = u.firstName;
                token.lastName = u.lastName;
                token.email = user.email;
                token.birthDate = u.birthDate;
                token.gender = u.gender;
                token.createdAt = u.createdAt;
                token.updatedAt = u.updatedAt;
                token.image = u.image;
            }
            if (account) {
                token.accessToken = account.access_token;
            }
            return token;
        },
        async session({ session, token }) {
            if (token && session.user) {
                session.user.id = token.id as string;
                const user = session.user as unknown as MatcherUser;
                user.firstName = token.firstName as string;
                user.lastName = token.lastName as string;
                user.birthDate = token.birthDate as string;
                user.gender = token.gender as string | null;
                user.createdAt = token.createdAt as string;
                user.updatedAt = token.updatedAt as string;
                user.image = token.image as string | null;
            }
            return session;
        },
    },

    pages: {},

    trustHost: true,
});
