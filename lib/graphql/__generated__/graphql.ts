export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
export type MakeEmpty<T extends { [key: string]: unknown }, K extends keyof T> = { [_ in K]?: never };
export type Incremental<T> = T | { [P in keyof T]?: P extends ' $fragmentName' | '__typename' ? T[P] : never };
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: { input: string; output: string; }
  String: { input: string; output: string; }
  Boolean: { input: boolean; output: boolean; }
  Int: { input: number; output: number; }
  Float: { input: number; output: number; }
};

export type Conversation = {
  __typename: 'Conversation';
  createdAt: Scalars['String']['output'];
  id: Scalars['ID']['output'];
  lastMessage: Maybe<Message>;
  lastMessageAt: Maybe<Scalars['String']['output']>;
  otherParticipant: User;
  participant1: User;
  participant2: User;
  unreadCount: Maybe<Scalars['Int']['output']>;
  updatedAt: Scalars['String']['output'];
};

export type CreateSpaceInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  joinPolicy?: InputMaybe<Scalars['String']['input']>;
  name: Scalars['String']['input'];
  slug?: InputMaybe<Scalars['String']['input']>;
  visibility?: InputMaybe<Scalars['String']['input']>;
};

export type CreateTierInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  interval: Scalars['String']['input'];
  name: Scalars['String']['input'];
  price: Scalars['Int']['input'];
};

export type CreateUserInput = {
  birthDate: Scalars['String']['input'];
  email: Scalars['String']['input'];
  firstName: Scalars['String']['input'];
  gender?: InputMaybe<Gender>;
  lastName: Scalars['String']['input'];
};

export enum Gender {
  Man = 'man',
  NonBinary = 'non_binary',
  Woman = 'woman'
}

export type MatchOptions = {
  gender?: InputMaybe<Array<Gender>>;
  limit?: InputMaybe<Scalars['Int']['input']>;
  maxAge?: InputMaybe<Scalars['Int']['input']>;
  minAge?: InputMaybe<Scalars['Int']['input']>;
};

export type Member = {
  __typename: 'Member';
  currentPeriodEnd: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  joinedAt: Scalars['String']['output'];
  role: Scalars['String']['output'];
  status: Scalars['String']['output'];
  subscriptionId: Maybe<Scalars['String']['output']>;
  tier: Maybe<MembershipTier>;
  user: User;
};

export type MembershipTier = {
  __typename: 'MembershipTier';
  currency: Scalars['String']['output'];
  description: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  interval: Scalars['String']['output'];
  isActive: Scalars['Boolean']['output'];
  name: Scalars['String']['output'];
  price: Scalars['Int']['output'];
  spaceId: Scalars['ID']['output'];
};

export type Message = {
  __typename: 'Message';
  content: Scalars['String']['output'];
  conversationId: Scalars['ID']['output'];
  createdAt: Scalars['String']['output'];
  id: Scalars['ID']['output'];
  readAt: Maybe<Scalars['String']['output']>;
  sender: User;
};

export type Mutation = {
  __typename: 'Mutation';
  approveMember: Member;
  archiveTier: Scalars['Boolean']['output'];
  createPost: Post;
  createSpace: Space;
  createTier: MembershipTier;
  createUser: User;
  deletePost: Scalars['Boolean']['output'];
  deleteSpace: Scalars['Boolean']['output'];
  deleteUser: Scalars['Boolean']['output'];
  joinSpace: Member;
  leaveSpace: Scalars['Boolean']['output'];
  markAsRead: Maybe<Scalars['Boolean']['output']>;
  removeMember: Scalars['Boolean']['output'];
  sendMessage: Message;
  startConversation: Conversation;
  updateMemberRole: Member;
  updateSpace: Space;
  updateTier: MembershipTier;
  updateUser: Maybe<User>;
};


export type MutationApproveMemberArgs = {
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationArchiveTierArgs = {
  id: Scalars['ID']['input'];
};


export type MutationCreatePostArgs = {
  content: Scalars['String']['input'];
  mediaUrls?: InputMaybe<Array<Scalars['String']['input']>>;
  spaceId: Scalars['ID']['input'];
};


export type MutationCreateSpaceArgs = {
  input: CreateSpaceInput;
};


export type MutationCreateTierArgs = {
  input: CreateTierInput;
  spaceId: Scalars['ID']['input'];
};


export type MutationCreateUserArgs = {
  input: CreateUserInput;
};


export type MutationDeletePostArgs = {
  postId: Scalars['ID']['input'];
};


export type MutationDeleteSpaceArgs = {
  id: Scalars['ID']['input'];
};


export type MutationDeleteUserArgs = {
  id: Scalars['ID']['input'];
};


export type MutationJoinSpaceArgs = {
  spaceSlug: Scalars['String']['input'];
  tierId?: InputMaybe<Scalars['ID']['input']>;
};


export type MutationLeaveSpaceArgs = {
  spaceId: Scalars['ID']['input'];
};


export type MutationMarkAsReadArgs = {
  conversationId: Scalars['ID']['input'];
};


export type MutationRemoveMemberArgs = {
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationSendMessageArgs = {
  content: Scalars['String']['input'];
  conversationId: Scalars['ID']['input'];
};


export type MutationStartConversationArgs = {
  targetUserId: Scalars['ID']['input'];
};


export type MutationUpdateMemberRoleArgs = {
  role: Scalars['String']['input'];
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
};


export type MutationUpdateSpaceArgs = {
  id: Scalars['ID']['input'];
  input: UpdateSpaceInput;
};


export type MutationUpdateTierArgs = {
  id: Scalars['ID']['input'];
  input: UpdateTierInput;
};


export type MutationUpdateUserArgs = {
  id: Scalars['ID']['input'];
  input: UpdateUserInput;
};

export type Post = {
  __typename: 'Post';
  author: User;
  commentsCount: Maybe<Scalars['Int']['output']>;
  content: Scalars['String']['output'];
  createdAt: Scalars['String']['output'];
  id: Scalars['ID']['output'];
  likesCount: Maybe<Scalars['Int']['output']>;
  mediaUrls: Maybe<Array<Scalars['String']['output']>>;
};

/** Profilo utente con traits e dati per matching */
export type Profile = {
  __typename: 'Profile';
  behavioralDescription: Maybe<Scalars['String']['output']>;
  createdAt: Scalars['String']['output'];
  embeddingsComputedAt: Maybe<Scalars['String']['output']>;
  id: Scalars['ID']['output'];
  interestsDescription: Maybe<Scalars['String']['output']>;
  psychologicalDescription: Maybe<Scalars['String']['output']>;
  updatedAt: Scalars['String']['output'];
  userId: Scalars['ID']['output'];
  valuesDescription: Maybe<Scalars['String']['output']>;
};

export type Query = {
  __typename: 'Query';
  conversation: Maybe<Conversation>;
  conversations: Array<Conversation>;
  /** Get daily suggested matches for the current user */
  dailyMatches: Array<User>;
  findMatches: Array<User>;
  globalFeed: Maybe<Array<Post>>;
  me: Maybe<User>;
  messages: Array<Message>;
  mySpaces: Array<Space>;
  space: Maybe<Space>;
  spaces: Array<Space>;
  user: Maybe<User>;
  users: Array<User>;
};


export type QueryConversationArgs = {
  id: Scalars['ID']['input'];
};


export type QueryFindMatchesArgs = {
  options?: InputMaybe<MatchOptions>;
  userId?: InputMaybe<Scalars['ID']['input']>;
};


export type QueryGlobalFeedArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type QueryMessagesArgs = {
  conversationId: Scalars['ID']['input'];
};


export type QuerySpaceArgs = {
  id?: InputMaybe<Scalars['ID']['input']>;
  slug?: InputMaybe<Scalars['String']['input']>;
};


export type QueryUserArgs = {
  id: Scalars['ID']['input'];
};

export type Space = {
  __typename: 'Space';
  clientId: Maybe<Scalars['String']['output']>;
  createdAt: Scalars['String']['output'];
  description: Maybe<Scalars['String']['output']>;
  feed: Maybe<Array<Post>>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  isActive: Maybe<Scalars['Boolean']['output']>;
  joinPolicy: Scalars['String']['output'];
  members: Maybe<Array<Member>>;
  membersCount: Maybe<Scalars['Int']['output']>;
  myMembership: Maybe<Member>;
  name: Scalars['String']['output'];
  slug: Scalars['String']['output'];
  tiers: Maybe<Array<MembershipTier>>;
  type: Maybe<Scalars['String']['output']>;
  visibility: Scalars['String']['output'];
};


export type SpaceFeedArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};


export type SpaceMembersArgs = {
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
};

export type UpdateSpaceInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  image?: InputMaybe<Scalars['String']['input']>;
  joinPolicy?: InputMaybe<Scalars['String']['input']>;
  name?: InputMaybe<Scalars['String']['input']>;
  visibility?: InputMaybe<Scalars['String']['input']>;
};

export type UpdateTierInput = {
  description?: InputMaybe<Scalars['String']['input']>;
  interval?: InputMaybe<Scalars['String']['input']>;
  isActive?: InputMaybe<Scalars['Boolean']['input']>;
  name?: InputMaybe<Scalars['String']['input']>;
  price?: InputMaybe<Scalars['Int']['input']>;
};

export type UpdateUserInput = {
  birthDate?: InputMaybe<Scalars['String']['input']>;
  email?: InputMaybe<Scalars['String']['input']>;
  firstName?: InputMaybe<Scalars['String']['input']>;
  gender?: InputMaybe<Gender>;
  lastName?: InputMaybe<Scalars['String']['input']>;
};

/** Utente base - dati anagrafici */
export type User = {
  __typename: 'User';
  birthDate: Scalars['String']['output'];
  createdAt: Scalars['String']['output'];
  email: Scalars['String']['output'];
  firstName: Scalars['String']['output'];
  gender: Maybe<Gender>;
  id: Scalars['ID']['output'];
  image: Maybe<Scalars['String']['output']>;
  lastName: Scalars['String']['output'];
  profile: Maybe<Profile>;
  updatedAt: Scalars['String']['output'];
};

export type ConversationFieldsFragment = { __typename: 'Conversation', id: string, lastMessageAt: string | null, createdAt: string, updatedAt: string, unreadCount: number | null, otherParticipant: { __typename: 'User', id: string, firstName: string, lastName: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: string } | null };

export type MessageFieldsFragment = { __typename: 'Message', id: string, conversationId: string, content: string, readAt: string | null, createdAt: string, sender: { __typename: 'User', id: string, firstName: string, lastName: string, image: string | null } };

export type GetConversationsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetConversationsQuery = { conversations: Array<{ __typename: 'Conversation', id: string, lastMessageAt: string | null, createdAt: string, updatedAt: string, unreadCount: number | null, otherParticipant: { __typename: 'User', id: string, firstName: string, lastName: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: string } | null }> };

export type GetRecentConversationsQueryVariables = Exact<{ [key: string]: never; }>;


export type GetRecentConversationsQuery = { conversations: Array<{ __typename: 'Conversation', id: string, unreadCount: number | null, otherParticipant: { __typename: 'User', firstName: string, lastName: string } }> };

export type GetMessagesQueryVariables = Exact<{
  conversationId: Scalars['ID']['input'];
}>;


export type GetMessagesQuery = { messages: Array<{ __typename: 'Message', id: string, conversationId: string, content: string, readAt: string | null, createdAt: string, sender: { __typename: 'User', id: string, firstName: string, lastName: string, image: string | null } }>, conversation: { __typename: 'Conversation', id: string, otherParticipant: { __typename: 'User', id: string, firstName: string, lastName: string, image: string | null } } | null };

export type StartConversationMutationVariables = Exact<{
  targetUserId: Scalars['ID']['input'];
}>;


export type StartConversationMutation = { startConversation: { __typename: 'Conversation', id: string, lastMessageAt: string | null, createdAt: string, updatedAt: string, unreadCount: number | null, otherParticipant: { __typename: 'User', id: string, firstName: string, lastName: string, image: string | null }, lastMessage: { __typename: 'Message', content: string, createdAt: string } | null } };

export type SendMessageMutationVariables = Exact<{
  conversationId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
}>;


export type SendMessageMutation = { sendMessage: { __typename: 'Message', id: string, content: string, createdAt: string } };

export type MarkAsReadMutationVariables = Exact<{
  conversationId: Scalars['ID']['input'];
}>;


export type MarkAsReadMutation = { markAsRead: boolean | null };

export type GetDailyMatchesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetDailyMatchesQuery = { dailyMatches: Array<{ __typename: 'User', id: string, firstName: string, lastName: string, email: string, birthDate: string, gender: Gender | null, image: string | null, createdAt: string, updatedAt: string }> };

export type UpdateMemberRoleMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
  role: Scalars['String']['input'];
}>;


export type UpdateMemberRoleMutation = { updateMemberRole: { __typename: 'Member', id: string, role: string } };

export type RemoveMemberMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  userId: Scalars['ID']['input'];
}>;


export type RemoveMemberMutation = { removeMember: boolean };

export type PostFieldsFragment = { __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, likesCount: number | null, commentsCount: number | null, createdAt: string, author: { __typename: 'User', id: string, firstName: string, lastName: string } };

export type GetGlobalFeedQueryVariables = Exact<{
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetGlobalFeedQuery = { globalFeed: Array<{ __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, likesCount: number | null, commentsCount: number | null, createdAt: string, author: { __typename: 'User', id: string, firstName: string, lastName: string } }> | null };

export type GetSpaceFeedQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  limit?: InputMaybe<Scalars['Int']['input']>;
  offset?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceFeedQuery = { space: { __typename: 'Space', feed: Array<{ __typename: 'Post', id: string, content: string, mediaUrls: Array<string> | null, likesCount: number | null, commentsCount: number | null, createdAt: string, author: { __typename: 'User', id: string, firstName: string, lastName: string } }> | null } | null };

export type CreatePostMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  content: Scalars['String']['input'];
  mediaUrls?: InputMaybe<Array<Scalars['String']['input']> | Scalars['String']['input']>;
}>;


export type CreatePostMutation = { createPost: { __typename: 'Post', id: string } };

export type DeletePostMutationVariables = Exact<{
  postId: Scalars['ID']['input'];
}>;


export type DeletePostMutation = { deletePost: boolean };

export type SpaceFieldsFragment = { __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, visibility: string, joinPolicy: string, createdAt: string, clientId: string | null, isActive: boolean | null, membersCount: number | null, type: string | null };

export type GetAllSpacesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetAllSpacesQuery = { spaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, visibility: string, joinPolicy: string, createdAt: string, clientId: string | null, isActive: boolean | null, membersCount: number | null, type: string | null }> };

export type GetMySpacesQueryVariables = Exact<{ [key: string]: never; }>;


export type GetMySpacesQuery = { mySpaces: Array<{ __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, visibility: string, joinPolicy: string, createdAt: string, clientId: string | null, isActive: boolean | null, membersCount: number | null, type: string | null }> };

export type GetSpaceQueryVariables = Exact<{
  id?: InputMaybe<Scalars['ID']['input']>;
  slug?: InputMaybe<Scalars['String']['input']>;
  membersLimit?: InputMaybe<Scalars['Int']['input']>;
}>;


export type GetSpaceQuery = { space: { __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, visibility: string, joinPolicy: string, createdAt: string, clientId: string | null, isActive: boolean | null, membersCount: number | null, type: string | null, myMembership: { __typename: 'Member', id: string, role: string, status: string, tier: { __typename: 'MembershipTier', id: string, name: string, price: number, interval: string } | null } | null, tiers: Array<{ __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: string, isActive: boolean, spaceId: string }> | null, members: Array<{ __typename: 'Member', id: string, role: string, status: string, joinedAt: string, tier: { __typename: 'MembershipTier', name: string } | null, user: { __typename: 'User', id: string, firstName: string, lastName: string, email: string } }> | null } | null };

export type CreateSpaceMutationVariables = Exact<{
  input: CreateSpaceInput;
}>;


export type CreateSpaceMutation = { createSpace: { __typename: 'Space', id: string, name: string, slug: string } };

export type UpdateSpaceMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateSpaceInput;
}>;


export type UpdateSpaceMutation = { updateSpace: { __typename: 'Space', id: string, name: string, slug: string, description: string | null, image: string | null, visibility: string, joinPolicy: string, createdAt: string, clientId: string | null, isActive: boolean | null, membersCount: number | null, type: string | null } };

export type DeleteSpaceMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type DeleteSpaceMutation = { deleteSpace: boolean };

export type JoinSpaceMutationVariables = Exact<{
  spaceSlug: Scalars['String']['input'];
  tierId?: InputMaybe<Scalars['ID']['input']>;
}>;


export type JoinSpaceMutation = { joinSpace: { __typename: 'Member', id: string, status: string } };

export type LeaveSpaceMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type LeaveSpaceMutation = { leaveSpace: boolean };

export type TierFieldsFragment = { __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: string, isActive: boolean, spaceId: string };

export type GetSpaceTiersQueryVariables = Exact<{
  spaceId: Scalars['ID']['input'];
}>;


export type GetSpaceTiersQuery = { space: { __typename: 'Space', id: string, tiers: Array<{ __typename: 'MembershipTier', id: string, name: string, description: string | null, price: number, currency: string, interval: string, isActive: boolean, spaceId: string }> | null } | null };

export type CreateTierMutationVariables = Exact<{
  spaceId: Scalars['ID']['input'];
  input: CreateTierInput;
}>;


export type CreateTierMutation = { createTier: { __typename: 'MembershipTier', id: string } };

export type ArchiveTierMutationVariables = Exact<{
  id: Scalars['ID']['input'];
}>;


export type ArchiveTierMutation = { archiveTier: boolean };

export type UserFieldsFragment = { __typename: 'User', id: string, firstName: string, lastName: string, email: string, birthDate: string, gender: Gender | null, image: string | null, createdAt: string, updatedAt: string };

export type GetMeQueryVariables = Exact<{ [key: string]: never; }>;


export type GetMeQuery = { me: { __typename: 'User', id: string, firstName: string, lastName: string, email: string, birthDate: string, gender: Gender | null, image: string | null, createdAt: string, updatedAt: string } | null };

export type UpdateUserMutationVariables = Exact<{
  id: Scalars['ID']['input'];
  input: UpdateUserInput;
}>;


export type UpdateUserMutation = { updateUser: { __typename: 'User', id: string, firstName: string, lastName: string, email: string, birthDate: string, gender: Gender | null, image: string | null, createdAt: string, updatedAt: string } | null };
